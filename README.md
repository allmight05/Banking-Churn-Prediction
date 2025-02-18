```
Bank Churn Prediction using Automated ML Pipelines

Project Overview

This project is designed to predict customer churn risk for a bank using machine learning. It automates data ingestion, validation, transformation, model training, and prediction through a well-structured pipeline. The risk of churn is classified on a scale of 1 to 5, where 1 represents the lowest risk and 5 represents the highest.

Project Structure

The project follows a modular structure, ensuring scalability and maintainability:

src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── util.py
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── logger.py
│   ├── exceptions/
│   │   ├── __init__.py
│   │   ├── exception.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   ├── constants/
│   │   ├── __init__.py
│   │   ├── constant.py

Key Features

Automated ML Pipelines: Handles the entire ML workflow, from data ingestion to prediction.

Data Processing Modules:

Data Ingestion: Loads and preprocesses raw data.

Data Validation: Ensures data quality and consistency.

Data Transformation: Prepares features for model training.

Model Training: Trains and evaluates machine learning models.

Prediction Pipeline: Uses trained models to predict customer churn risk.

Logging and Exception Handling: Ensures smooth debugging and monitoring.

Installation

Clone the repository:

git clone <repository_url>
cd <project_directory>

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

Usage

Train the Model

Run the training pipeline to train the machine learning model:

python src/pipeline/train_pipeline.py

Make Predictions

Run the prediction pipeline to get churn risk scores:

python src/pipeline/predict_pipeline.py

Technologies Used

Python

Pandas, NumPy (Data Processing)

Scikit-learn (Machine Learning)

TensorFlow/PyTorch (Optional for Deep Learning)

Logging and Exception Handling Modules

License

This project is licensed under the MIT License.
