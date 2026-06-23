import os
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self):
        try:
            logging.info("Starting Data Transformation")
            
            df = pd.read_csv(self.config.data_path)
            logging.info(f"Loaded data for transformation with shape: {df.shape}")

            # Features to scale
            features = ['Recency', 'Frequency', 'Monetary']
            
            # Handling potential outliers in RFM (often needed for K-Means)
            # Using log transformation to handle skewness
            df_log = df.copy()
            for col in features:
                # Add 1 to handle zero values
                df_log[col] = np.log1p(df[col])

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_log[features])
            
            # Combine CustomerID and scaled features
            scaled_df = pd.DataFrame(scaled_features, columns=features)
            scaled_df.insert(0, 'CustomerID', df['CustomerID'])

            # Split data into train, validation, and test sets
            # First split: train and temp (test + val)
            train_df, temp_df = train_test_split(scaled_df, test_size=0.3, random_state=42)
            
            # Second split: validation and test
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

            # Save the preprocessor
            joblib.dump(scaler, self.config.preprocessor_path)
            logging.info(f"Scaler saved to {self.config.preprocessor_path}")

            # Save the transformed data splits
            np.save(self.config.train_data_path, train_df.to_numpy())
            np.save(self.config.val_data_path, val_df.to_numpy())
            np.save(self.config.test_data_path, test_df.to_numpy())
            
            logging.info(f"Train data saved to {self.config.train_data_path} with shape {train_df.shape}")
            logging.info(f"Val data saved to {self.config.val_data_path} with shape {val_df.shape}")
            logging.info(f"Test data saved to {self.config.test_data_path} with shape {test_df.shape}")

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise e
