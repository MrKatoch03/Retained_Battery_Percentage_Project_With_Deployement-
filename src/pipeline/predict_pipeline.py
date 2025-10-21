import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    """
    Prediction pipeline for battery capacity retention
    Loads trained model and preprocessor, then make predictions
    """
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.power_transformer = None

    def predict(self, features):
        """
        Make predictions on input features

        Args: 
            features: DataFrame with battery features
        Returns:
            predictions: Predicted capacity retained percentage 
        """
        try:
            logging.info("Starting prediction pipeline")

            # FIX: Use relative paths that work in both Windows and Linux
            # Get the base directory (project root)
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            model_path = os.path.join(base_dir, 'artifacts', 'model.pkl')
            label_encoders_path = os.path.join(base_dir, 'artifacts', 'label_encoder.pkl')
            power_transformer_path = os.path.join(base_dir, 'artifacts', 'power_transformer.pkl')

            # Debug logging
            logging.info(f"Base directory: {base_dir}")
            logging.info(f"Model path: {model_path}")
            logging.info(f"Model exists: {os.path.exists(model_path)}")

            # Load all preprocessors
            self.model = load_object(file_path=model_path)
            self.label_encoder = load_object(file_path=label_encoders_path)
            self.power_transformer = load_object(file_path=power_transformer_path)

            logging.info("Model and Preprocessor loaded successfully")

            # Apply label encoding to categorical columns
            logging.info("Apply label encoding to categorical features...")
            categorical_cols = ['manufacturer', 'chemistry']
            features_encoded = features.copy()

            for col in categorical_cols:
                if col in features_encoded.columns:
                    features_encoded[col + '_encoded'] = self.label_encoder[col].transform(
                        features_encoded[col].astype(str)
                    )

            # Drop original categorical columns
            features_encoded = features_encoded.drop(categorical_cols, axis=1)
            logging.info("Label encoding completed")

            # Apply power transformation to numeric features
            numeric_cols = [
                'capacity_kWh',
                'charge_cycles',
                'avg_temp_celsius',
                'discharge_rate_c',
                'charge_rate_c',
                'avg_soc_percent',
                'storage_time_months',
                'fast_charge_ratio',
                'calendar_age_years'
            ]

            # Apply transformation
            features_transformed = self.power_transformer.transform(features_encoded[numeric_cols])

            # Create DataFrame with transformed features
            features_transformed_df = pd.DataFrame(
                features_transformed,
                columns=numeric_cols,
                index=features_encoded.index
            )

            # Combine with encoded categorical features
            encoded_cols = [col + '_encoded' for col in categorical_cols]
            X_final = pd.concat([features_transformed_df, features_encoded[encoded_cols]], axis=1)

            logging.info("Power Transformation Completed")

            # Make predictions
            logging.info("Making predictions..")
            predictions = self.model.predict(X_final.values)

            logging.info(f"Predictions completed: {predictions}")

            return predictions
            
        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise CustomException(e, sys)


class BatteryData:
    """
    Custom data class for battery capacity predictions
    Converts user input into DataFrame format for predictions 
    """

    def __init__(self,
                 manufacturer: str,
                 chemistry: str,
                 capacity_kWh: float,
                 charge_cycles: int,
                 avg_temp_celsius: float,
                 discharge_rate_c: float,
                 charge_rate_c: float,
                 avg_soc_percent: float,
                 storage_time_months: int, 
                 fast_charge_ratio: float,
                 calendar_age_years: float):
        """
        Initialize BatteryData with battery parameters
        
        Args:
            manufacturer: Battery manufacturer (Tesla, Panasonic, LG Chem, CATL, BYD, Samsung SDI)
            chemistry: Battery chemistry (NMC, LFP, NCA)
            capacity_kWh: Nominal battery capacity in kWh (40-120)
            charge_cycles: Number of full charge-discharge cycles (50-3000)
            avg_temp_celsius: Average operating temperature in °C (-10 to 50)
            discharge_rate_c: Typical discharge rate in C (0.5-5)
            charge_rate_c: Typical charge rate in C (0.5-3)
            avg_soc_percent: Average state of charge during use (10-90%)
            storage_time_months: Months in storage before use (0-24)
            fast_charge_ratio: Fraction of fast charging events (0-1)
            calendar_age_years: Age since manufacturing (0.1-12 years)
        """
        self.manufacturer = manufacturer
        self.chemistry = chemistry
        self.capacity_kWh = capacity_kWh
        self.charge_cycles = charge_cycles
        self.avg_temp_celsius = avg_temp_celsius
        self.discharge_rate_c = discharge_rate_c
        self.charge_rate_c = charge_rate_c
        self.avg_soc_percent = avg_soc_percent
        self.storage_time_months = storage_time_months
        self.fast_charge_ratio = fast_charge_ratio
        self.calendar_age_years = calendar_age_years
        
    def get_data_as_data_frame(self):
        """
        Convert battery data into pandas DataFrame format
        
        Returns:
            DataFrame: Single row DataFrame with battery features
        """
        try:
            logging.info("Converting battery data to DataFrame")
            
            battery_data_dict = {
                "manufacturer": [self.manufacturer],
                "chemistry": [self.chemistry],
                "capacity_kWh": [self.capacity_kWh],
                "charge_cycles": [self.charge_cycles],
                "avg_temp_celsius": [self.avg_temp_celsius],
                "discharge_rate_c": [self.discharge_rate_c],
                "charge_rate_c": [self.charge_rate_c],
                "avg_soc_percent": [self.avg_soc_percent],
                "storage_time_months": [self.storage_time_months],
                "fast_charge_ratio": [self.fast_charge_ratio],
                "calendar_age_years": [self.calendar_age_years],
            }
            
            df = pd.DataFrame(battery_data_dict)
            logging.info("✓ DataFrame created successfully")
            logging.info(f"\nBattery Data:\n{df.to_string(index=False)}")
            
            return df
        
        except Exception as e:
            logging.error(f"Error converting battery data to DataFrame: {str(e)}")
            raise CustomException(e, sys)


# Example usage
if __name__ == "__main__":
    try:
        logging.info("=" * 80)
        logging.info("BATTERY CAPACITY PREDICTION")
        logging.info("=" * 80)
        
        # Create sample battery data
        battery_data = BatteryData(
            manufacturer="Tesla",
            chemistry="NMC",
            capacity_kWh=75.0,
            charge_cycles=1500,
            avg_temp_celsius=25.0,
            discharge_rate_c=2.5,
            charge_rate_c=1.5,
            avg_soc_percent=50.0,
            storage_time_months=6,
            fast_charge_ratio=0.3,
            calendar_age_years=3.5
        )
        
        # Convert to DataFrame
        input_df = battery_data.get_data_as_data_frame()
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(input_df)
        
        logging.info("=" * 80)
        logging.info("PREDICTION RESULT")
        logging.info("=" * 80)
        logging.info(f"\nPredicted Capacity Retained: {prediction[0]:.2f}%")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise