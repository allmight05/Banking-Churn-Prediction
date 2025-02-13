import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exceptions.exception import ProjectException
from src.logging import logger


@dataclass
class DataValidationConfig:
    valid_train_data_path: str = os.path.join('validated', 'train_valid_data.csv')
    valid_test_data_path: str = os.path.join('validated', 'test_valid_data.csv')


@dataclass
class DataValidationArtifact:
    valid_train_data_path: str
    valid_test_data_path: str
    validation_status: bool = True

class DataValidation:
    def __init__(self, train_data_path: str, test_data_path: str):
        """
        Initializes the data validation module with the paths to the train and test data.
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.validation_config = DataValidationConfig()
        self.validation_artifact = None

    def initiate_data_validation(self):
        logger.logging.info("Data Validation started")
        try:
            
            if not os.path.exists(self.train_data_path):
                raise ProjectException(f"Train data file not found at {self.train_data_path}", sys)
            train_df = pd.read_csv(self.train_data_path)
            logger.logging.info("Train data read successfully for validation")
            
            # Drop rows with missing values in 'region_category' and force a copy.
            train_df_cleaned = train_df.dropna(subset=['region_category']).copy()
            
            # Fill missing values in 'points_in_wallet' with the column mean using .loc.
            if 'points_in_wallet' in train_df_cleaned.columns:
                mean_points = train_df_cleaned['points_in_wallet'].mean()
                train_df_cleaned.loc[:, 'points_in_wallet'] = train_df_cleaned['points_in_wallet'].fillna(mean_points)
            
            # Fill missing values in 'preferred_offer_types' with the column mode using .loc.
            if 'preferred_offer_types' in train_df_cleaned.columns:
                mode_offer = train_df_cleaned['preferred_offer_types'].mode()[0]
                train_df_cleaned.loc[:, 'preferred_offer_types'] = train_df_cleaned['preferred_offer_types'].fillna(mode_offer)
            
            # Convert 'avg_frequency_login_days' to float32 if it exists.
            if 'avg_frequency_login_days' in train_df_cleaned.columns:
                train_df_cleaned.loc[:, 'avg_frequency_login_days'] = pd.to_numeric(train_df_cleaned['avg_frequency_login_days'], errors='coerce')
                
                mean_login_train_days = train_df_cleaned['avg_frequency_login_days'].mean()
                train_df_cleaned['avg_frequency_login_days']=train_df_cleaned['avg_frequency_login_days'].fillna(mean_login_train_days)
            
            # Drop unnecessary columns if they exist.
            cols_to_drop = ['customer_id', 'Name', 'security_no', 'referral_id']
            train_df_cleaned = train_df_cleaned.drop(columns=cols_to_drop, errors='ignore')
            
            
            if not os.path.exists(self.test_data_path):
                raise ProjectException(f"Test data file not found at {self.test_data_path}", sys)
            test_df = pd.read_csv(self.test_data_path)
            logger.logging.info("Test data read successfully for validation")
            
            # Drop rows with missing values in 'region_category' and force a copy.
            test_df_cleaned = test_df.dropna(subset=['region_category']).copy()
            
            # Fill missing values in 'points_in_wallet' with the column mean using .loc.
            if 'points_in_wallet' in test_df_cleaned.columns:
                mean_points_test = test_df_cleaned['points_in_wallet'].mean()
                test_df_cleaned.loc[:, 'points_in_wallet'] = test_df_cleaned['points_in_wallet'].fillna(mean_points_test)
            
            # Fill missing values in 'preferred_offer_types' with the column mode using .loc.
            if 'preferred_offer_types' in test_df_cleaned.columns:
                mode_offer_test = test_df_cleaned['preferred_offer_types'].mode()[0]
                test_df_cleaned.loc[:, 'preferred_offer_types'] = test_df_cleaned['preferred_offer_types'].fillna(mode_offer_test)
            
            # Convert 'avg_frequency_login_days' to float32 if it exists.
            if 'avg_frequency_login_days' in test_df_cleaned.columns:
                test_df_cleaned.loc[:, 'avg_frequency_login_days'] = pd.to_numeric(test_df_cleaned['avg_frequency_login_days'], errors='coerce')
                mean_login_test_days = test_df_cleaned['avg_frequency_login_days'].mean()
                test_df_cleaned['avg_frequency_login_days']=test_df_cleaned['avg_frequency_login_days'].fillna(mean_login_test_days)
            
            # Drop unnecessary columns if they exist.
            test_df_cleaned = test_df_cleaned.drop(columns=cols_to_drop, errors='ignore')
            
            # # After cleaning both train and test dataframes
            # if train_df_cleaned.isnull().any().any():
            #     missing_cols_train = train_df_cleaned.columns[train_df_cleaned.isnull().any()].tolist()
            #     logger.logging.warning(f"Train data still has missing values in columns: {missing_cols_train}")

            # if test_df_cleaned.isnull().any().any():
            #     missing_cols_test = test_df_cleaned.columns[test_df_cleaned.isnull().any()].tolist()
            #     logger.logging.warning(f"Test data still has missing values in columns: {missing_cols_test}")

            
            os.makedirs(os.path.dirname(self.validation_config.valid_train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.validation_config.valid_test_data_path), exist_ok=True)

            train_df_cleaned.to_csv(self.validation_config.valid_train_data_path, index=False)
            test_df_cleaned.to_csv(self.validation_config.valid_test_data_path, index=False)

            logger.logging.info("Data Validation completed successfully")

            # Prepare the artifact with validation details.
            self.validation_artifact = DataValidationArtifact(
                valid_train_data_path=self.validation_config.valid_train_data_path,
                valid_test_data_path=self.validation_config.valid_test_data_path,
                validation_status=True
            )

            return self.validation_artifact

        except Exception as e:
            raise ProjectException(e, sys)

# if __name__ == "__main__":
#     # Assume that the data ingestion step has already created these files.
#     train_data_path = os.path.join("artifacts", "train.csv")
#     test_data_path = os.path.join("artifacts", "test.csv")
#     validator = DataValidation(train_data_path, test_data_path)
#     artifact = validator.initiate_data_validation()
#     print("Validation Artifact:", artifact)



