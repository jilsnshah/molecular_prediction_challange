# Molecular Prediction Challenge

Machine learning project for predicting molecular dipole moments using SOAP descriptors and XGBoost.

## Project Structure

```
├── .gitignore                 # Git ignore configuration
├── ml_optimized.ipynb         # Main ML notebook with SOAP + Coulomb Matrix + XGBoost
├── only_baysen.ipynb          # Bayesian XGBoost approach
├── another_method.ipynb       # Alternative experimental methods
├── train.csv                  # Training data with dipole moments
├── test.csv                   # Test data for predictions
├── submission.csv             # Submission file
├── structures_train/          # Training molecular structures (20,000 .xyz files)
├── structures_test/           # Test molecular structures (5,000 .xyz files)
├── xgb_model10.json          # Trained XGBoost model
├── xgb_model11.json          # Trained XGBoost model (variant)
└── lifi/                      # LiFi transmission project (separate)
```

## Features

- **SOAP Descriptors**: Smooth Overlap of Atomic Positions for molecular feature extraction
- **Coulomb Matrix**: Additional molecular descriptor
- **XGBoost**: Gradient boosting for regression
- **Bayesian Optimization**: Hyperparameter tuning

## Requirements

- Python 3.10+
- dscribe
- ase
- xgboost
- scikit-learn
- pandas
- numpy
- tqdm

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open and run `ml_optimized.ipynb` for the main pipeline
4. Check `only_baysen.ipynb` for Bayesian approach

## Models

The project uses XGBoost regression with:
- SOAP descriptors (r_cut=5.0, n_max=8, l_max=6)
- Coulomb Matrix descriptors
- Feature combination strategies

## LiFi Project

The `lifi/` folder contains a separate project for light-based data transmission using Arduino and LDR sensors.

## License

MIT

## Author

jilsnshah
