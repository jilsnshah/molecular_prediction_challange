# Molecular Prediction Challenge

Machine learning project for predicting molecular dipole moments using various approaches including SOAP descriptors with XGBoost, standard Neural Networks, and Graph Neural Networks (GNNs).

## Project Structure

```text
├── .gitignore                 # Git ignore configuration
├── dataset.py                 # Dataset loader for standard NN
├── model.py                   # Standard Neural Network model
├── train.py                   # Training script for standard NN
├── predict.py                 # Prediction script for standard NN
├── gnn_dataset.py             # Dataset loader for GNN
├── gnn_model.py               # Graph Neural Network model
├── gnn_train.py               # Training script for GNN
├── gnn_predict.py             # Prediction script for GNN
├── ml_optimized.ipynb         # Main ML notebook with SOAP + Coulomb Matrix + XGBoost
├── only_baysen.ipynb          # Bayesian XGBoost approach
├── another_method.ipynb       # Alternative experimental methods
├── train.csv                  # Training data with dipole moments
├── test.csv                   # Test data for predictions
├── structures_train/          # Training molecular structures (20,000 .xyz files)
├── structures_test/           # Test molecular structures (5,000 .xyz files)
├── best_model.pth             # Trained standard NN model weights
├── best_gnn_model.pth         # Trained GNN model weights
├── xgb_model10.json           # Trained XGBoost model
└── xgb_model11.json           # Trained XGBoost model (variant)
```

## Features

- **SOAP Descriptors**: Smooth Overlap of Atomic Positions for molecular feature extraction
- **Coulomb Matrix**: Additional molecular descriptor
- **XGBoost**: Gradient boosting for regression
- **Neural Networks (PyTorch)**: Custom dataset loading, training, and prediction for molecular features
- **Graph Neural Networks (PyTorch Geometric)**: Advanced molecular property prediction using graph representations
- **Bayesian Optimization**: Hyperparameter tuning

## Requirements

- Python 3.10+
- PyTorch
- PyTorch Geometric (for GNN)
- dscribe
- ase
- xgboost
- scikit-learn
- pandas
- numpy
- tqdm

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt` (Note: ensure PyTorch and PyTorch Geometric are installed)
3. Open and run `ml_optimized.ipynb` for the XGBoost pipeline
4. Use `python train.py` and `python predict.py` for standard Neural Networks
5. Use `python gnn_train.py` and `python gnn_predict.py` for Graph Neural Networks

## Models

The project explores multiple architectures:
- XGBoost regression with SOAP and Coulomb Matrix descriptors
- Standard Feedforward Neural Networks
- Graph Neural Networks (GNN) for directly learning from molecular topology

## License

MIT

## Author

jilsnshah
