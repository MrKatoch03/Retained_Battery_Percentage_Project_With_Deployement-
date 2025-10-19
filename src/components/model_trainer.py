import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    model_report_file_path = os.path.join("artifacts", "model_report.txt")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Main training orchestration function
        """
        try:
            logging.info("=" * 80)
            logging.info("MODEL TRAINING INITIATED")
            logging.info("=" * 80)
            
            logging.info("Splitting training and test input data")
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_test shape: {y_test.shape}")
            
            # Define models with hyperparameters for GridSearchCV
            models = {
                'Linear Regression': {
                    'model': LinearRegression(),
                    'params': {
                        'fit_intercept': [True, False]
                    }
                },
                'Decision Tree': {
                    'model': DecisionTreeRegressor(random_state=42),
                    'params': {
                        'max_depth': [5, 10, 15, 20, None],
                        # 'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'Random Forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        # 'max_depth': [10, 20, 30, None],
                        # 'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                },
                'K-Neighbors': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': [3, 5, 7, 9, 11],
                        # 'weights': ['uniform', 'distance'],
                        'p': [1, 2]
                    }
                },
                'XGBoost': {
                    'model': XGBRegressor(random_state=42, verbosity=0),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        # 'max_depth': [3, 5, 7],
                        # 'learning_rate': [0.01, 0.1, 0.3],
                        'subsample': [0.8, 1.0]
                    }
                },
                'CatBoost': {
                    'model': CatBoostRegressor(random_state=42, verbose=0),
                    'params': {
                        'iterations': [100, 200, 300],
                        # 'depth': [4, 6, 8],
                        'learning_rate': [0.01, 0.1, 0.3]
                    }
                },
                'AdaBoost': {
                    'model': AdaBoostRegressor(random_state=42),
                    'params': {
                        # 'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 1.0]
                    }
                }
            }
            
            logging.info("=" * 80)
            logging.info("GRID SEARCH EVALUATION STARTED")
            logging.info("=" * 80)
            
            # Evaluate all models using utils function
            results, model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            
            logging.info("=" * 80)
            logging.info("MODEL EVALUATION COMPLETED")
            logging.info("=" * 80)
            
            # Convert results to DataFrame for better visualization
            if not results:
                raise CustomException("No results returned from model evaluation", sys)
            
            results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Best_Model'} 
                                      for r in results])
            
            # Sort by Test R2 Score (descending)
            results_df = results_df.sort_values('Test_R2', ascending=False)
            
            logging.info("\nDetailed Results (sorted by Test R² - descending):")
            logging.info("\n" + results_df.to_string(index=False))
            
            # Get best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = [r['Best_Model'] for r in results if r['Model'] == best_model_name][0]
            best_result = results_df.iloc[0]
            
            logging.info("=" * 80)
            logging.info("BEST MODEL SUMMARY")
            logging.info("=" * 80)
            logging.info(f"\n🏆 Best Model: {best_model_name}")
            logging.info(f"   Test R² Score: {best_result['Test_R2']:.4f}")
            logging.info(f"   Test RMSE: {best_result['Test_RMSE']:.4f}")
            logging.info(f"   Test MAE: {best_result['Test_MAE']:.4f}")
            logging.info(f"   Overfit Score: {best_result['Overfit']:.4f}")
            logging.info(f"\n   Best Parameters: {best_result['Best_Params']}")
            
            # Check if best model score meets threshold
            if best_model_score < 0.6:
                logging.warning(f"Best model R² score ({best_model_score:.4f}) is below threshold (0.6)")
                logging.info("Proceeding with best available model...")
            
            logging.info(f"\n✓ Best model found on both training and testing dataset")
            
            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"✓ Best model saved at: {self.model_trainer_config.trained_model_file_path}")
            
            # Save model report as text file
            with open(self.model_trainer_config.model_report_file_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MODEL TRAINING REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("DETAILED RESULTS (sorted by Test R² - descending):\n")
                f.write("-" * 80 + "\n")
                f.write(results_df.to_string(index=False))
                f.write("\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("BEST MODEL SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Best Model: {best_model_name}\n")
                f.write(f"Test R² Score: {best_result['Test_R2']:.4f}\n")
                f.write(f"Test RMSE: {best_result['Test_RMSE']:.4f}\n")
                f.write(f"Test MAE: {best_result['Test_MAE']:.4f}\n")
                f.write(f"Overfit Score: {best_result['Overfit']:.4f}\n")
                f.write(f"\nBest Parameters:\n")
                for param, value in best_result['Best_Params'].items():
                    f.write(f"  - {param}: {value}\n")
            
            logging.info(f"✓ Model report saved at: {self.model_trainer_config.model_report_file_path}")
            
            # Calculate final R² score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            logging.info("=" * 80)
            logging.info(f"FINAL TEST R² SCORE: {r2_square:.4f}")
            logging.info("=" * 80)
            
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)

            