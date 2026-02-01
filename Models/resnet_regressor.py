import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

print("Loading data...")
# Load all datasets
epri = pd.read_csv("mv-data/epri.csv")
hcb = pd.read_csv("mv-data/hcb.csv")
ieeeall = pd.read_csv("mv-data/ieeeall.csv")

# Combine datasets
data = pd.concat([epri, hcb, ieeeall], ignore_index=True)
print(f"Total samples: {len(data)}")

# Select relevant features for prediction
# Ones we use: Iameas (arc current), t (time), D_mm (distance), gap_mm, Voc (voltage)
feature_cols = ['Iameas', 't', 'D_mm', 'gap_mm', 'Voc', 'Ibf']
target_col = 'IEmeas'  # Incident energy measured

print("\nCleaning data...")
# Filter data with all required columns present
data_clean = data.dropna(subset=feature_cols + [target_col])
print(f"After removing NaN: {len(data_clean)} samples")

# Remove rows where target is zero or any feature is zero 
print("Removing zero values...")
data_clean = data_clean[(data_clean[target_col] > 0)]
for col in feature_cols:
    data_clean = data_clean[(data_clean[col] > 0)]
print(f"After removing zeros: {len(data_clean)} samples")

# Remove outliers (helps with learning)
print("Removing extreme outliers...")
Q1 = data_clean[target_col].quantile(0.01)
Q3 = data_clean[target_col].quantile(0.99)
data_clean = data_clean[(data_clean[target_col] >= Q1) & (data_clean[target_col] <= Q3)]
print(f"Final clean samples: {len(data_clean)}")

X = data_clean[feature_cols].values
y = data_clean[target_col].values.reshape(-1, 1)

print(f"\nTarget statistics:")
print(f"  Min: {y.min():.2f}, Max: {y.max():.2f}")
print(f"  Mean: {y.mean():.2f}, Median: {np.median(y):.2f}")

# Scale features
print("\nScaling features...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

print(f"Feature shape: {X_scaled.shape}")
print(f"Target shape: {y_scaled.shape}")


# ResNet Block
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Skip connection
        if in_features != out_features:
            self.skip = nn.Linear(in_features, out_features)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


# ResNet Regressor
class ResNetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 256, 256, 128], dropout=0.2):
        super(ResNetRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Initial layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Residual blocks
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(prev_dim if prev_dim != input_dim else hidden_dims[0], 
                                       hidden_dim, dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# Custom Dataset
class ArcFlashDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Loss Functions
def rmsle_loss(y_pred, y_true):
    """Root Mean Squared Logarithmic Error"""
    # Clip to avoid log of negative or zero values
    y_pred_clipped = torch.clamp(y_pred, min=1e-6)
    y_true_clipped = torch.clamp(y_true, min=1e-6)
    
    log_diff = torch.log(y_pred_clipped + 1) - torch.log(y_true_clipped + 1)
    return torch.sqrt(torch.mean(log_diff ** 2))

def mse_loss(y_pred, y_true):
    """Mean Squared Error"""
    return torch.mean((y_pred - y_true) ** 2)


# Prepare dataset
dataset = ArcFlashDataset(X_scaled, y_scaled)

# Split into train/validation/test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"\nTrain: {train_size}, Val: {val_size}, Test: {test_size}")

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\nBuilding model...")
model = ResNetRegressor(input_dim=X_scaled.shape[1], 
                        hidden_dims=[128, 256, 256, 128],
                        dropout=0.3).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                  factor=0.5, patience=10)

# Training loop
print("\n" + "="*60)
print("Training ResNet Regressor...")
print("="*60)
num_epochs = 250
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = mse_loss(y_pred, y_batch)  # Use MSE for better learning
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = mse_loss(y_pred, y_batch)  # Use MSE for better learning
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'resnet_best_model.pth')
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

print("\n" + "="*60)
print("Training completed!")
print("="*60)

# Save scalers
print("\nSaving model artifacts...")
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✓ Scalers saved: scaler_X.pkl, scaler_y.pkl")

# Load best model for evaluation
print("\nLoading best model for evaluation...")
checkpoint = torch.load('resnet_best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best model loaded from epoch {checkpoint['epoch']+1}")

# Evaluate on test set
print("Evaluating on test set...")
model.eval()
test_predictions = []
test_actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred = model(X_batch)
        test_predictions.extend(y_pred.cpu().numpy())
        test_actuals.extend(y_batch.numpy())

test_predictions = np.array(test_predictions)
test_actuals = np.array(test_actuals)

# Inverse transform to original scale
test_predictions_orig = scaler_y.inverse_transform(test_predictions)
test_actuals_orig = scaler_y.inverse_transform(test_actuals)

# Calculate metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
test_rmsle = np.sqrt(np.mean((np.log(test_predictions_orig + 1) - np.log(test_actuals_orig + 1)) ** 2))
r2 = r2_score(test_actuals_orig, test_predictions_orig)
mae = mean_absolute_error(test_actuals_orig, test_predictions_orig)
rmse = np.sqrt(mean_squared_error(test_actuals_orig, test_predictions_orig))

print("\n" + "="*60)
print("Test Set Performance:")
print("="*60)
print(f"RMSLE:     {test_rmsle:.4f}")
print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:.4f}")
print(f"RMSE:      {rmse:.4f}")
print("="*60)

# Plotting
print("\nGenerating plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training history
axes[0].plot(train_losses, label='Train MSE', alpha=0.8)
axes[0].plot(val_losses, label='Val MSE', alpha=0.8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training History')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Predicted vs Actual (log scale)
axes[1].scatter(test_actuals_orig, test_predictions_orig, alpha=0.6, s=30)
min_val = min(test_actuals_orig.min(), test_predictions_orig.min())
max_val = max(test_actuals_orig.max(), test_predictions_orig.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
axes[1].set_xlabel('Actual Incident Energy (cal/cm²)')
axes[1].set_ylabel('Predicted Incident Energy (cal/cm²)')
axes[1].set_title(f'Predicted vs Actual\nTest RMSLE: {test_rmsle:.4f}, R²: {r2:.4f}')
axes[1].legend()
axes[1].grid(True)
axes[1].set_xscale('log')
axes[1].set_yscale('log')

# Plot 3: Residuals
residuals = test_predictions_orig - test_actuals_orig
axes[2].scatter(test_actuals_orig, residuals, alpha=0.6, s=30)
axes[2].axhline(y=0, color='r', linestyle='--', lw=2)
axes[2].set_xlabel('Actual Incident Energy (cal/cm²)')
axes[2].set_ylabel('Residual (Predicted - Actual)')
axes[2].set_title('Residual Plot')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('resnet_predictions.png', dpi=300, bbox_inches='tight')
print("\nPlot saved: resnet_predictions.png")

# Additional linear scale plot
fig2, ax = plt.subplots(figsize=(8, 8))
ax.scatter(test_actuals_orig, test_predictions_orig, alpha=0.6, s=30, c='blue', edgecolors='k', linewidth=0.5)
min_val = min(test_actuals_orig.min(), test_predictions_orig.min())
max_val = max(test_actuals_orig.max(), test_predictions_orig.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('Actual Incident Energy (cal/cm²)', fontsize=12)
ax.set_ylabel('Predicted Incident Energy (cal/cm²)', fontsize=12)
ax.set_title(f'ResNet Regressor: Predicted vs Actual Incident Energy\nTest RMSLE: {test_rmsle:.4f}, R²: {r2:.4f}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('resnet_predictions_linear.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: resnet_predictions_linear.png")

print("\n" + "="*60)
print("MODEL TRAINING AND EVALUATION COMPLETE!")
print("="*60)
print(f"✓ Model saved: resnet_best_model.pth")
print(f"✓ Scalers saved: scaler_X.pkl, scaler_y.pkl")
print(f"✓ Plots saved: resnet_predictions.png, resnet_predictions_linear.png")
print("="*60)

plt.show()
