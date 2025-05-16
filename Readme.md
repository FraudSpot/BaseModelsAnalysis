# Fraud Detection System

A machine learning-based fraud detection system initially developed during the [SGH x Mastercard Hackathon (May 2025)](https://www.kaggle.com/competitions/sgh-x-mastercard-hackathon-may-2025/overview) and being advanced toward production-ready MVP status.

## Overview

This project implements an advanced fraud detection system for financial transactions using machine learning techniques. The system analyzes transaction data combined with merchant and user information to identify potentially fraudulent activities while minimizing false positives.

## Architecture

The system architecture follows a modular approach:

```
├── Model_base.py           # Core model implementation with advanced ML capabilities
├── transactions.json       # Transaction dataset (500k records)
├── merchants.csv           # Merchant information (1k records)
├── users.csv               # User data (20k records)
└── notebooks/              
    ├── Base_Implementation_XGBoost_training_results.ipynb     # Initial implementation
    └── Additional_finetuned_result.ipynb                      # Advanced model tuning
```

### Data Flow

1. **Data Ingestion**: Transaction, merchant, and user data are loaded and merged
2. **Preprocessing**: Feature engineering, handling missing values, encoding
3. **Model Training**: Training with class imbalance handling
4. **Evaluation**: Comprehensive metrics calculation
5. **Deployment**: Model persistence and API development (planned)

## Model Implementation

The core model implementation (`Model_base.py`) provides:

- Multiple classifier options (RandomForest, XGBoost, LightGBM)
- Advanced feature engineering
- Hyperparameter optimization
- Class imbalance handling techniques
- Cross-validation with stratification
- Feature selection
- Model persistence

## Key Features

- **Robust Evaluation Metrics**: Focus on precision-recall balance suitable for fraud detection
- **Feature Importance Analysis**: Identification of key fraud indicators
- **Threshold Optimization**: Adjustable decision thresholds to balance false positives/negatives
- **Cross-Validation**: Stratified validation to handle imbalanced data
- **Model Persistence**: Save/load functionality for trained models

## Performance

The model achieves strong performance on the test dataset:
- High recall for fraud detection
- Improved precision through advanced feature engineering and model tuning
- Good ROC AUC scores through hyperparameter optimization


## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- xgboost/lightgbm (optional for advanced models)
- imbalanced-learn (for handling class imbalance)
- joblib (for model persistence)

## Usage

Basic usage example:

```python
from Model_base import FraudDetectionModel

# Initialize model
model = FraudDetectionModel(n_estimators=200, max_depth=10)

# Train model
model.fit(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(metrics)

# Make predictions
predictions = model.predict(X_new)

# Save model for later use
model.save_model()
```



## Acknowledgements

This project was initially developed during the SGH x Mastercard Hackathon. We would like to thank the organizers for providing the challenge and dataset that made this project possible.
