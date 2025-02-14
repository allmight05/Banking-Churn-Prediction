import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exceptions.exception import ProjectException
from src.logging.logger import logging
from src.utils.util import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        """
        Expects train_array and test_array where the last column contains the churn risk labels
        (an integer from 1 to 5).
        """
        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Internally map labels from [1,2,3,4,5] to [0,1,2,3,4]
            y_train_mapped = y_train - 1
            y_test_mapped = y_test - 1
            
            # Define a dictionary of classification models.
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                # "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                # "AdaBoost Classifier": AdaBoostClassifier(),
            }
            
            # Define hyperparameter grids for each classifier.
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [50, 100, 150, 200, 250, 300, 350]
                },
                "Logistic Regression": {},
                "K-Neighbors Classifier": {},
                "XGBClassifier": {
                    'learning_rate': [0.1,0.01,0.05, 0.005, 0.001],
                    'n_estimators': [50, 100, 150, 200, 250]
                },
                # # "CatBoosting Classifier": {
                # #     'depth': [6, 8, 10],
                # #     'learning_rate': [0.01, 0.05, 0.1],
                # #     'iterations': [30, 50, 100]
                # },
                # "AdaBoost Classifier": {
                #     'learning_rate': [0.1, 0.01, 0.5, 0.001],
                #     'n_estimators': [8, 16, 32, 64, 128, 256]
                # }
            }
            
            # Evaluate models on the mapped labels
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train_mapped,
                X_test=X_test,
                y_test=y_test_mapped,
                models=models,
                param=params
            )
            
            best_model_score = max(model_report.values())
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise ProjectException('No best model found with sufficient accuracy')
            
            logging.info(f'Best model found: {best_model_name} with accuracy: {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            
            # Convert the predicted labels back to the original format ([1,2,3,4,5])
            predicted_original = predicted + 1
            
            accuracy = accuracy_score(y_test, predicted_original)
            return accuracy
            
        except Exception as e:
            raise ProjectException(e, sys)


