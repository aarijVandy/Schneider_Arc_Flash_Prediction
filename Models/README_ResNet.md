# ResNet Regressor for Arc Flash Incident Energy Prediction

This ResNet-based deep learning model predicts incident energy from arc flash parameters.

## Features
- **ResNet Architecture**: Deep residual network with skip connections for better gradient flow
- **RMSLE Optimization**: Optimized for Root Mean Squared Logarithmic Error
- **Robust Training**: Includes batch normalization, dropout, and learning rate scheduling
- **Comprehensive Evaluation**: Generates prediction plots and residual analysis

## Input Features
The model takes 6 input parameters:
1. `Iameas` - Arc current (kA)
2. `t` - Time duration (seconds)
3. `D_mm` - Distance (mm)
4. `gap_mm` - Electrode gap (mm)
5. `Voc` - Open circuit voltage (kV)
6. `Ibf` - Bolted fault current (kA)

## Output
- `IEmeas` - Incident energy (cal/cm²)

## Installation

First, install the required dependencies:

```bash
cd Models
bash install_dependencies.sh
```

Or if that doesnt work, simply:
```bash
pip install torch torchvision torchaudio scikit-learn joblib pandas matplotlib numpy
```

## Training

```bash
python resnet_regressor.py
```

This saves the best model to `resnet_best_model.pth` -- change this later

## Model Files

After training, you'll have:
- `resnet_best_model.pth` - Trained model weights
- `scaler_X.pkl` - Feature scaler
- `scaler_y.pkl` - Target scaler
- `resnet_predictions.png` - Log-scale prediction plots


## Model Architecture

```
Input (6 features)
    ↓
Linear + BatchNorm + ReLU + Dropout
    ↓
ResidualBlock (128)
    ↓
ResidualBlock (256)
    ↓
ResidualBlock (256)
    ↓
ResidualBlock (128)
    ↓
Linear (1 output)
```


## Performance Metrics

The model reports:
- **RMSLE** (Root Mean Squared Logarithmic Error) - Primary metric
- **R² Score** - Coefficient of determination
- Training history plots
- Predicted vs Actual scatter plots (log and linear scales)
- Residual analysis

## Notes

- The model uses 70/15/15 train/validation/test split
- implemtent early stopping based on validation loss
- Batch size: 32
- Dropout: 0.3 --- change later
- Weight decay: 1e-5 for L2 regularization
