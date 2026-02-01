import torch
import joblib
import numpy as np
import pandas as pd

### 
# 
# Main script to use to run the tests. this runs it once 
#
###


print("Loading saved model and scalers...")

# Load scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Load model 
from resnet_regressor import ResNetRegressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetRegressor(input_dim=6, hidden_dims=[128, 256, 256, 128], dropout=0.3).to(device)

# Load trained weights
checkpoint = torch.load('resnet_best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

print("Model loaded successfully!")
print(f"Best epoch: {checkpoint['epoch']+1}")
print(f"Validation RMSLE: {checkpoint['val_loss']:.4f}")


# Example input: [Iameas, t, D_mm, gap_mm, Voc, Ibf]
example_input = np.array([[
    15.0,   # Iameas (arc current in kA)
    0.5,    # t (time in seconds)
    914.0,  # D_mm (distance in mm)
    32.0,   # gap_mm (electrode gap in mm)
    13.8,   # Voc (open circuit voltage in kV)
    20.0    # Ibf (bolted fault current in kA)
]])

print("\nInput parameters:")
print(f"  Arc Current (Iameas): {example_input[0][0]} kA")
print(f"  Time (t): {example_input[0][1]} seconds")
print(f"  Distance (D_mm): {example_input[0][2]} mm")
print(f"  Gap (gap_mm): {example_input[0][3]} mm")
print(f"  Voltage (Voc): {example_input[0][4]} kV")
print(f"  Fault Current (Ibf): {example_input[0][5]} kA")

# Scale input
example_scaled = scaler_X.transform(example_input)

# Make inference
with torch.no_grad():
    example_tensor = torch.FloatTensor(example_scaled).to(device)
    prediction_scaled = model(example_tensor).cpu().numpy()

# Inverse transform
prediction = scaler_y.inverse_transform(prediction_scaled)

print(f"\nPredicted Incident Energy: {prediction[0][0]:.2f} cal/cm²")
print("="*60)
