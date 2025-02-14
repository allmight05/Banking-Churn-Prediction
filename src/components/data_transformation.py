import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Import your custom exception and logger modules
from src.exceptions.exception import ProjectException
from src.logging import logger
from src.utils.util import save_object  # function to pickle objects

#############################################
# Custom Transformer for Data Cleaning and Feature Engineering
#############################################

def categorize_time(time_obj):
    """Categorize a time object into parts of day."""
    hour = time_obj.hour
    if 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'


class CustomDataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that performs the following:
      - Replaces -999 in days_since_last_login with the median (computed on valid values).
      - Clips outliers in avg_time_spent, avg_frequency_login_days and points_in_wallet 
        using the 25th and 75th percentiles.
      - Encodes 'feedback' and 'membership_category' using provided mappings.
      - Processes 'last_visit_time' into a categorical feature.
      - Computes days_since_joined from 'joining_date'.
    """
    def __init__(self, current_date_str='2024-07-29',
                 clip_lower_quantile=0.25, clip_upper_quantile=0.75):
        self.current_date = pd.to_datetime(current_date_str)
        self.clip_lower_quantile = clip_lower_quantile
        self.clip_upper_quantile = clip_upper_quantile
        # These will be computed in fit()
        self.dlt_login_median_ = None
        self.clip_dict_ = {}
        # Mappings provided in your instructions.
        self.feedback_encodings = {
            "Quality Customer Care": 0, "Products always in Stock": 0, 
            "User Friendly Website": 0, "Reasonable Price": 0, 
            "Poor Product Quality": 1, "No Membership": 1,
            "Poor Website": 1, "No reason specified": 1,
            "Too many ads": 1
        }
        self.membership_encodings = {
            "Platinum Membership": 0, "Premium Membership": 0, 
            "Gold Membership": 1, "Silver Membership": 1, 
            "Basic Membership": 2, "No Membership": 2
        }

    def fit(self, X, y=None):
        try:
            df = X.copy()
            # For days_since_last_login, compute the median from valid values (i.e. not -999)
            valid_days = df.loc[df['days_since_last_login'] != -999, 'days_since_last_login']
            self.dlt_login_median_ = valid_days.median()

            # For outlier clipping: compute lower and upper quantiles for specified numerical columns
            outlier_cols = ['avg_time_spent', 'avg_frequency_login_days', 'points_in_wallet']
            for col in outlier_cols:
                lower = df[col].quantile(self.clip_lower_quantile)
                upper = df[col].quantile(self.clip_upper_quantile)
                self.clip_dict_[col] = (lower, upper)

            return self

        except Exception as e:
            raise ProjectException(e, sys)

    def transform(self, X, y=None):
        try:
            df = X.copy()

            # Replace -999 in days_since_last_login with computed median.
            if 'days_since_last_login' in df.columns:
                df['days_since_last_login'] = df['days_since_last_login'].replace(-999, self.dlt_login_median_)

            # Clip outliers for specified columns.
            for col, (lower, upper) in self.clip_dict_.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower, upper)

            # Encode 'feedback' column.
            if 'feedback' in df.columns:
                df['feedback'] = df['feedback'].map(self.feedback_encodings)
                # Fill missing values from mapping with default value (assumed 1)
                df['feedback'] = df['feedback'].fillna(1)

            # Encode 'membership_category' column.
            if 'membership_category' in df.columns:
                df['membership_category'] = df['membership_category'].map(self.membership_encodings)
                # Fill missing values from mapping with default value (assumed 2)
                df['membership_category'] = df['membership_category'].fillna(2)

            # Process 'last_visit_time': convert to datetime.time and create a category.
            if 'last_visit_time' in df.columns:
                df['last_visit_time'] = pd.to_datetime(df['last_visit_time'], format='%H:%M:%S', errors='coerce').dt.time
                df['last_visit_time_category'] = df['last_visit_time'].apply(
                    lambda t: categorize_time(t) if pd.notnull(t) else 'Unknown'
                )
                df.drop(columns=['last_visit_time'], inplace=True)

            # Process 'joining_date': convert to datetime and compute days_since_joined.
            if 'joining_date' in df.columns:
                df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
                df['days_since_joined'] = (self.current_date - df['joining_date']).dt.days
                # Fill missing days_since_joined with median if necessary
                df['days_since_joined'] = df['days_since_joined'].fillna(df['days_since_joined'].median())
                df.drop(columns=['joining_date'], inplace=True)

            return df

        except Exception as e:
            raise ProjectException(e, sys)

#############################################
# Data Transformation Module
#############################################

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self, train_data_path: str, test_data_path: str, target_column: str = None):
        """
        Initializes the data transformation module.
        If target_column is provided, it will be separated from the features.
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.target_column = target_column
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns the full preprocessing pipeline:
          - CustomDataFrameTransformer: performs custom cleaning and feature engineering.
          - ColumnTransformer: applies imputation, scaling to numerical features and one-hot encoding to categorical features.
        """
        try:
            # Updated list of numerical features - remove target column if present.
            numerical_features = [
                'age', 'days_since_last_login', 'avg_time_spent',
                'avg_transaction_value', 'avg_frequency_login_days',
                'points_in_wallet', 'membership_category', 'feedback', 'days_since_joined'
            ]

            # Categorical features: remaining categorical features after our custom processing.
            categorical_features = [
                'gender', 'region_category', 'joined_through_referral',
                'preferred_offer_types', 'medium_of_operation',
                'internet_option', 'used_special_discount',
                'offer_application_preference', 'past_complaint',
                'complaint_status', 'last_visit_time_category'
            ]

            # Pipeline for numerical features: impute missing values with median and then scale.
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Pipeline for categorical features: impute missing values with a constant and then one-hot encode.
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine the numerical and categorical pipelines into a ColumnTransformer.
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])

            # Create the full pipeline by first applying the custom transformer then the ColumnTransformer.
            full_pipeline = Pipeline(steps=[
                ('custom_transform', CustomDataFrameTransformer()),
                ('column_transformer', preprocessor)
            ])

            logger.logging.info("Data transformation pipeline created successfully.")
            return full_pipeline

        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_data_transformation(self):
        """
        Reads the train and test data, applies the transformation pipeline, saves the preprocessor,
        and returns the transformed train and test arrays along with the path of the saved preprocessor.
        """
        try:
            # Read train data.
            train_df = pd.read_csv(self.train_data_path)
            # Remove unnamed columns
            train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
            logger.logging.info("Train data read successfully for transformation.")

            # Read test data.
            test_df = pd.read_csv(self.test_data_path)
            # Remove unnamed columns
            test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]
            logger.logging.info("Test data read successfully for transformation.")

            # Filter out rows where the target value is -1, if target_column is provided.
            if self.target_column and self.target_column in train_df.columns:
                train_df = train_df[train_df[self.target_column] != -1].reset_index(drop=True)
            if self.target_column and self.target_column in test_df.columns:
                test_df = test_df[test_df[self.target_column] != -1].reset_index(drop=True)

            # Separate the target column (if provided) from the features.
            if self.target_column and self.target_column in train_df.columns:
                X_train = train_df.drop(columns=[self.target_column])
                y_train = train_df[self.target_column]
            else:
                X_train = train_df.copy()
                y_train = None

            if self.target_column and self.target_column in test_df.columns:
                X_test = test_df.drop(columns=[self.target_column])
                y_test = test_df[self.target_column]
            else:
                X_test = test_df.copy()
                y_test = None

            # Obtain the preprocessing pipeline and fit-transform on training data; transform test data.
            preprocessing_pipeline = self.get_data_transformer_object()

            X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
            X_test_transformed = preprocessing_pipeline.transform(X_test)

            # If target exists, concatenate the transformed features with the target.
            if y_train is not None:
                train_arr = np.c_[X_train_transformed, np.array(y_train)]
            else:
                train_arr = X_train_transformed

            if y_test is not None:
                test_arr = np.c_[X_test_transformed, np.array(y_test)]
            else:
                test_arr = X_test_transformed

            # Save the preprocessor object.
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_pipeline
            )
            logger.logging.info("Preprocessing pipeline saved successfully.")

            return train_arr, test_arr, self.transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise ProjectException(e, sys)


#############################################
# For Testing / Execution
#############################################

# if __name__ == "__main__":
#     try:
#         # Example file paths; adjust as needed.
#         train_data_path = os.path.join('validated', 'train_valid_data.csv')
#         test_data_path = os.path.join('validated', 'test_valid_data.csv')
#         # Specify the target column which is churn_risk_score.
#         transformation_obj = DataTransformation(train_data_path, test_data_path, target_column="churn_risk_score")
#         train_arr, test_arr, preprocessor_path = transformation_obj.initiate_data_transformation()
#         logger.logging.info(f"Data Transformation completed. Preprocessor saved at: {preprocessor_path}")
#     except Exception as e:
#         logger.logging.error(f"Error in Data Transformation: {e}")
#         raise ProjectException(e, sys)
