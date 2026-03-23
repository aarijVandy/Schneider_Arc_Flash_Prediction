import torch
import joblib
import numpy as np

print("Loading saved model and scalers...")

feature_cols = ["t", "D_mm", "gap_mm", "Voc", "Ibf"]

# Load scalers
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

from resnet_regressor import LSTMRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("lstm_best_model.pth", map_location=device)
window_size = checkpoint.get("window_size", 10)
model = LSTMRegressor(
    input_dim=len(feature_cols),
    hidden_dim=128,
    num_layers=2,
    dropout=0.3,
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

print("Model loaded successfully!")
print(f"Best epoch: {checkpoint['epoch'] + 1}")
print(f"Validation MSE: {checkpoint['val_loss']:.4f}")
print(f"Window size: {window_size}")

# Example sequence with length = window_size
example_step = np.array([
    0.5,    # t (seconds)
    914.0,  # D_mm (mm)
    32.0,   # gap_mm (mm)
    13.8,   # Voc (kV)
    20.0,   # Ibf (kA)
])

example_sequence = np.tile(example_step, (window_size, 1))

print("\nExample sequence (repeated step):")
print(example_sequence)

# Scale and predict
example_scaled = scaler_X.transform(example_sequence)
example_scaled = example_scaled[np.newaxis, ...]  # add batch dim

with torch.no_grad():
    example_tensor = torch.FloatTensor(example_scaled).to(device)
    pred_scaled = model(example_tensor).cpu().numpy()

pred = scaler_y.inverse_transform(pred_scaled)

print(f"\nPredicted Incident Energy: {pred[0][0]:.2f} cal/cm²")
print("=" * 60)
