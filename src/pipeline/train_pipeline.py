import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logging import logger  
from src.exceptions.exception import ProjectException

def run_train_pipeline():
    try:
        logger.logging.info("========== Train Pipeline Started ==========")
        
        # 1. Data Ingestion
        logger.logging.info("Starting Data Ingestion...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initate_data_ingestion()
        logger.logging.info(f"Data Ingestion Completed. Train data at: {train_data_path}, Test data at: {test_data_path}")
        
        # 2. Data Validation
        logger.logging.info("Starting Data Validation...")
        data_validation = DataValidation(train_data_path=train_data_path, test_data_path=test_data_path)
        validation_artifact = data_validation.initiate_data_validation()
        logger.logging.info("Data Validation Completed.")
        
        # 3. Data Transformation
        logger.logging.info("Starting Data Transformation...")
        # For a classification task, assume the target column is 'churn_risk_score'
        data_transformation = DataTransformation(
            train_data_path=validation_artifact.valid_train_data_path,
            test_data_path=validation_artifact.valid_test_data_path,
            target_column="churn_risk_score"
        )
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
        logger.logging.info(f"Data Transformation Completed. Preprocessor saved at: {preprocessor_path}")
        
        # 4. Model Training (Classification)
        logger.logging.info("Starting Model Training...")
        model_trainer = ModelTrainer()
        # The modified model trainer returns accuracy for classification models.
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logger.logging.info(f"Model Training Completed. Accuracy: {accuracy:.4f}")
        
        logger.logging.info("========== Train Pipeline Completed Successfully ==========")
        
    except Exception as e:
        logger.logging.error(f"Error in Train Pipeline: {e}")
        raise ProjectException(e, sys)

if __name__ == "__main__":
    run_train_pipeline()
