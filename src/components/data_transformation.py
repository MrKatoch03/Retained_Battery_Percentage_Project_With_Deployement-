import sys 
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.ensemble import IsolationForest

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os 

# To store object 
@dataclass
class DataTransformationConfig:
    label_encoder_file_path = os.path.join('artifacts','label_encoder.pkl')
    power_transformer_file_path = os.path.join('artifacts','power_transformer.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.label_encoder={}
        self.power_transformer = None

    def drop_irrelevant_columns(self, df, drop_cols):
        """
        Step 0 : Drop Irrelevant columns 
        """
        try: 
            logging.info(f"Dropping irrelevant columns: {drop_cols}")
            df = df.drop(drop_cols,axis=1)
            logging.info(f"Columns dropped. Remaining shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomException(e,sys)

    def handle_missing_values_categorical(self, df, categorical_cols):
        """
        Step 1: Fill missing values in categorical columns with 'Unknown'
        """
        try: 
            logging.info(f"Handlling missing values in categorical columns")
            for col in categorical_cols:
                df[col] = df[col].fillna('Unknown')
            logging.info("Categorical missing values filled with 'Unknown'")
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def handling_missing_values_numerical(self,df,numerical_cols, target_col, groupby_cols):
        """
        Step 2: Fill numeric missing values with median grouped by categorical columns 
        """

        try:
            logging.info(f"Handling missing values in numeric columns: {numerical_cols}")
            logging.info(f"Using median grouped by: {groupby_cols}")


            df[numerical_cols] = df.groupby(groupby_cols)[numerical_cols].transform(lambda x: x.fillna(x.median()))

            logging.info("Numeric missing values filled with grouped median")
            logging.info(f"Null counts after grouped median fill:\n{df[numerical_cols].isnull().sum()}")
            if target_col in df.columns:
                target_median = df[target_col].median()
                df[target_col] = df[target_col].fillna(target_median)
                logging.info(f"Target column '{target_col}' missing values filled with overall median: {target_median}")
            
            logging.info(f"Null counts after median fill:\n{df[numerical_cols + [target_col]].isnull().sum()}")
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def encode_categorical_features(self, df, categorical_cols):
        """
        Step 3: Encode categorical columns using LabelEncoder and drop originals
        """
        try:
            logging.info(f"Encoding Categorical Columns: {categorical_cols}")

            self.label_encoder = {}

            for col in categorical_cols:
                le = LabelEncoder()

                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoder[col] = le
                logging.info(f"Encoded '{col}' -> '{col}_encoded'")

            df = df.drop(categorical_cols,axis = 1)
            logging.info(f"Dropped original categorical columns")
            logging.info(f"Null counts after encoding TEST data:\n{df.isnull().sum()}")


            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def remove_outliers_isolation_forest(self, df, contamination=0.08, n_estimator=120):
        """
        Step 4: Removes outliers using Isolation Forest on numeric columns 
        """

        try: 
            logging.info("Starting outlier removal using Isolation Forest")

            numeric_cols = df.select_dtypes(include=['number']).columns
            df_numeric = df[numeric_cols].copy()

            logging.info(f"Original dataset shape: {df.shape}")
            logging.info(f"Numeric columns for outlier detection: {len(numeric_cols)}")

            ##Handling any remaining nulls
            if df_numeric.isnull().any().any():
                logging.warning("Nulls detected in numeric columns. Filling with medians...")
                df_numeric = df_numeric.fillna(df_numeric.median())
            
            ##Initialize and fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=n_estimator

            )

            outlier_label  = iso_forest.fit_predict(df_numeric)

            ### Filter outliers (keep only inliners: label == 1)
            df_clean = df[outlier_label == 1].copy()

            outliers_removed = len(df) - len(df_clean)
            outliers_percentage = (outliers_removed / len(df))*100

            logging.info(f"Outlier detected and removed: {outliers_removed} ({outliers_percentage:.2f}%)")
            logging.info(f"Clean dataset shape: {df_clean.shape}")

            return df_clean
        except Exception as e:
            raise CustomException(e,sys)
    
    def normalize_features_power_transformer (self, df, target_col, encoded_cols,fit=False):
        """
        Step 5: Normalize numeric features using PowerTransformer (Yeo-Johnson)
        Separates target and encoded featutes before transformation
        """

        try: 
            logging.info("Starting feature normalization with PowerTransformer")

            #Separating components

            y = df[target_col].copy()
            columns_to_exclude = [target_col] + encoded_cols
            X = df.drop(columns_to_exclude, axis=1)
            X_encoded = df[encoded_cols].copy()

            logging.info(f"Features shape (X): {X.shape}")
            logging.info(f"Encoded features shape: {X_encoded.shape}")
            logging.info(f"Target shape (y):{y.shape}")

            #Select numeric columns 
            numeric_cols = X.select_dtypes(include=['number']).columns
            logging.info(f"Numeric columns to normalize: {list(numeric_cols)}")

            #check skewness before
            skewness_before = X[numeric_cols].skew().sort_values(ascending=False)
            logging.info(f"Skewness Before Normalization :\n{skewness_before}")

            if fit:

                logging.info("FITTING PowerTransformer on training data...")

                #Apply Yeo-Johnson transformation
                self.power_transformer = PowerTransformer(method='yeo-johnson',standardize=True)
                X_normalized = self.power_transformer.fit_transform(X[numeric_cols])
            else:
                logging.info("TRANSFORMING with pre-fitted PowerTransformer (trained on train data)...")
                X_normalized = self.power_transformer.transform(X[numeric_cols])

            # Convert back to DataFrame
            X_normalized_df = pd.DataFrame(
                X_normalized,
                columns = numeric_cols,
                index = X.index
            )

            # Concatenate with encoded features 
            X_final = pd.concat ([X_normalized_df,X_encoded],axis=1)

            #Check skewness after
            skewness_after = X_final[numeric_cols].skew().sort_values(ascending=False)
            logging.info(f"Skewness after normalization: \n {skewness_after}")

            logging.info(f"Normalization complete. Final Shape: {X_final.shape}")
            logging.info(f"Shapes match: {X_final.shape[0] == y.shape[0]}")

            return X_final, y

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path, drop_cols, categorical_cols, numeric_cols, target_col, groupby_cols):
        """
        Main orchestration function: applies all EDA steps in sequence
        """

        try: 
            #Load data
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info(f"Train shape: {train_df.shape}, Test Shape: {test_df.shape}")

            #Step 1: Handling categorical missing 
            logging.info("=" * 60)
            logging.info("STEP 0: Dropping irrelevant columns")
            logging.info("=" * 60)
            train_df = self.drop_irrelevant_columns(train_df, drop_cols)
            test_df = self.drop_irrelevant_columns(test_df, drop_cols)
            logging.info(f"After dropping columns - Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            #Step 1: Handle categorical missing values
            logging.info("=" * 60)
            logging.info("STEP 1: Handling categorical missing values")
            logging.info("=" * 60)
            train_df = self.handle_missing_values_categorical(train_df,categorical_cols)
            test_df = self.handle_missing_values_categorical(test_df,categorical_cols)

            #Step 2: Handle numeric missing values with grounded median 
            logging.info("=" * 60)
            logging.info("STEP 2: Handling numeric missing values with grouped median")
            logging.info("=" * 60)
            train_df = self.handling_missing_values_numerical(train_df, numeric_cols, target_col, groupby_cols)
            test_df = self.handling_missing_values_numerical(test_df, numeric_cols, target_col, groupby_cols)

            # Step 3: Encode categorical features
            logging.info("=" * 60)
            logging.info("STEP 3: Encoding categorical features")
            logging.info("=" * 60)
            logging.info(">>> FITTING LabelEncoders on TRAINING DATA <<<")
            train_df = self.encode_categorical_features(train_df, categorical_cols)
            logging.info(">>> APPLYING encoders to TEST DATA <<<")
            for col in categorical_cols:
                test_df[col+'_encoded'] = self.label_encoder[col].transform(test_df[col].astype(str))
            test_df = test_df.drop(categorical_cols,axis=1)

            logging.info(f"Null counts after encoding TEST data:\n{test_df.isnull().sum()}")


            #Save label encoders
            save_object(
                file_path=self.data_transformation_config.label_encoder_file_path, 
                obj=self.label_encoder
            )

            #Step 4: Remove outliers (train set only)
            logging.info("=" * 60)
            logging.info("STEP 4: Removing outliers using Isolation Forest")
            logging.info("=" * 60)
            train_df = self.remove_outliers_isolation_forest(train_df, contamination=0.05)

            # Step 5: Normalize features
            logging.info("=" * 60)
            logging.info("STEP 5: Normalizing features with PowerTransformer")
            logging.info("=" * 60)
            encoded_cols = [col+'_encoded' for col in categorical_cols]
            logging.info(">>> FITTING PowerTransformer on TRAINING DATA <<<")
            X_train, y_train = self.normalize_features_power_transformer(
                train_df, target_col, encoded_cols, fit=True 
            )
            logging.info(">>> TRANSFORMING TEST DATA using fitted transformer <<<")
            X_test, y_test = self.normalize_features_power_transformer(
                test_df, target_col, encoded_cols, fit=False
            )

            #Save power transformer 
            save_object(
                file_path = self.data_transformation_config.power_transformer_file_path,
                obj = self.power_transformer
            )

            #combine features and target
            train_arr = np.c_[X_train.values, y_train.values]
            test_arr = np.c_[X_test.values, y_test.values]

            logging.info("=" * 60)
            logging.info("DATA TRANSFORMATION COMPLETE")
            logging.info("=" * 60)
            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.label_encoder_file_path,
                self.data_transformation_config.power_transformer_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    categorical_cols = ['manufacturer','chemistry']
    train_path = r'D:\MLOPSS\Project 1\artifacts\train.csv'
    test_path = r'D:\MLOPSS\Project 1\artifacts\test.csv'
    drop_cols = ['Battery_id']

        











