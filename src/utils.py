import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save object to pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at: {file_path}")
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load object from pickle file
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logging.info(f"Object loaded successfully from: {file_path}")
        return obj
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models using GridSearchCV
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        models: Dictionary of models with hyperparameters
                Format: {
                    'Model_Name': {
                        'model': model_object,
                        'params': {param1: [values], param2: [values]}
                    }
                }
    
    Returns:
        results: List of result dictionaries with detailed metrics
        model_report: Dictionary with model names and test R² scores
    """
    try:
        logging.info("Starting model evaluation with GridSearchCV")
        
        results = []
        model_report = {}

        for model_name, model_info in models.items():
            logging.info(f"Training {model_name}...")
            
            try:
                # GridSearchCV
                grid_search = GridSearchCV(
                    estimator=model_info['model'],
                    param_grid=model_info['params'],
                    cv=5,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
            except Exception as e:
                raise CustomException(e,sys)

            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Store results
            result = {
                'Model': model_name,
                'Best_Params': grid_search.best_params_,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Overfit': train_r2 - test_r2,
                'Best_Model': best_model
            }
            
            results.append(result)
            model_report[model_name] = test_r2
            
            logging.info(f"✓ {model_name} - Test R² Score: {test_r2:.4f}")
            logging.info(f"  Best Parameters: {grid_search.best_params_}")
        
        return results, model_report
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model_basic(X_train, y_train, X_test, y_test, models):
    """
    Basic model evaluation without GridSearchCV (simple baseline)
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        models: Dictionary of model objects (simple)
                Format: {'Model_Name': model_object}
    
    Returns:
        report: Dictionary with model names and test R² scores
    """
    try:
        logging.info("Starting basic model evaluation")
        
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_r2
            
            logging.info(f"✓ {model_name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)