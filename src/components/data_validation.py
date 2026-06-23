import os
import pandas as pd
import logging
from src.entity.config_entity import DataValidationConfig
from src.constants import PROJECT_ROOT

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            # Hardcoding the expected columns for now, though usually loaded from schema.yaml
            expected_columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

            # Read the ingested RFM data produced by the data ingestion stage.
            # Resolved against PROJECT_ROOT so it works regardless of cwd.
            data_path = PROJECT_ROOT / "artifacts" / "data_ingestion" / "customer_data.csv"
            data = pd.read_csv(data_path)
            
            all_cols = list(data.columns)

            for col in all_cols:
                if col not in expected_columns:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    break
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            logging.error(f"Error in data validation: {e}")
            raise e
