import os
import pandas as pd
import logging
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logging.info("Started Data Ingestion process")
        try:
            # Read the raw Online Retail dataset
            # Note: The dataset contains non-ASCII characters, typically 'ISO-8859-1' encoding is used
            df = pd.read_csv(self.config.data_path, encoding='ISO-8859-1')
            logging.info(f"Loaded raw data with shape {df.shape}")

            # Basic cleaning for RFM
            # 1. Drop missing CustomerID
            df = df.dropna(subset=['CustomerID'])
            
            # 2. Drop cancelled orders (Quantity < 0) or zero price
            df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

            # 3. Create TotalAmount
            df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

            # 4. Convert InvoiceDate to datetime
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

            # RFM Calculation
            # Recency: Latest date in dataset + 1 day
            reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
            
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (reference_date - x.max()).days,
                'InvoiceNo': 'count',
                'TotalAmount': 'sum'
            }).reset_index()

            rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
            logging.info(f"Generated RFM features with shape {rfm.shape}")

            # Save the processed data
            rfm.to_csv(self.config.ingested_data_path, index=False)
            logging.info(f"Ingested RFM data saved to {self.config.ingested_data_path}")

        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise e
