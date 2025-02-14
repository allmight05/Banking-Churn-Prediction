import sys
import pandas as pd
from src.exceptions.exception import ProjectException
from src.utils.util import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise ProjectException(e, sys)


class CustomData:
    def __init__(
        self,
        age: int,
        gender: str,
        region_category: str,
        membership_category: str,
        joining_date: str,
        joined_through_referral: str,
        preferred_offer_types: str,
        medium_of_operation: str,
        internet_option: str,
        last_visit_time: str,
        days_since_last_login: int,
        avg_time_spent: float,
        avg_transaction_value: float,
        avg_frequency_login_days: float,
        points_in_wallet: float,
        used_special_discount: str,
        offer_application_preference: str,
        past_complaint: str,
        complaint_status: str,
        feedback: str,
        churn_risk_score: int
    ):
        self.age = age
        self.gender = gender
        self.region_category = region_category
        self.membership_category = membership_category
        self.joining_date = joining_date
        self.joined_through_referral = joined_through_referral
        self.preferred_offer_types = preferred_offer_types
        self.medium_of_operation = medium_of_operation
        self.internet_option = internet_option
        self.last_visit_time = last_visit_time
        self.days_since_last_login = days_since_last_login
        self.avg_time_spent = avg_time_spent
        self.avg_transaction_value = avg_transaction_value
        self.avg_frequency_login_days = avg_frequency_login_days
        self.points_in_wallet = points_in_wallet
        self.used_special_discount = used_special_discount
        self.offer_application_preference = offer_application_preference
        self.past_complaint = past_complaint
        self.complaint_status = complaint_status
        self.feedback = feedback
        self.churn_risk_score = churn_risk_score

    def get_data_as_data_frame(self):
        try:
            data_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "region_category": [self.region_category],
                "membership_category": [self.membership_category],
                "joining_date": [self.joining_date],
                "joined_through_referral": [self.joined_through_referral],
                "preferred_offer_types": [self.preferred_offer_types],
                "medium_of_operation": [self.medium_of_operation],
                "internet_option": [self.internet_option],
                "last_visit_time": [self.last_visit_time],
                "days_since_last_login": [self.days_since_last_login],
                "avg_time_spent": [self.avg_time_spent],
                "avg_transaction_value": [self.avg_transaction_value],
                "avg_frequency_login_days": [self.avg_frequency_login_days],
                "points_in_wallet": [self.points_in_wallet],
                "used_special_discount": [self.used_special_discount],
                "offer_application_preference": [self.offer_application_preference],
                "past_complaint": [self.past_complaint],
                "complaint_status": [self.complaint_status],
                "feedback": [self.feedback],
                "churn_risk_score": [self.churn_risk_score]
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise ProjectException(e, sys)
        
# input_data = CustomData(
#     age=25,
#     gender="Male",
#     region_category="Urban",
#     membership_category="Gold",
#     joining_date="2022-01-15",
#     joined_through_referral="Yes",
#     preferred_offer_types="Discount",
#     medium_of_operation="Online",
#     internet_option="WiFi",
#     last_visit_time="2022-12-01 10:00:00",
#     days_since_last_login=5,
#     avg_time_spent=15.5,
#     avg_transaction_value=100.0,
#     avg_frequency_login_days=3.5,
#     points_in_wallet=200.0,
#     used_special_discount="No",
#     offer_application_preference="Email",
#     past_complaint="No",
#     complaint_status="Resolved",
#     feedback="Positive",
#     churn_risk_score=2
# )
# df = input_data.get_data_as_data_frame()

# pipeline = PredictPipeline()
# prediction = pipeline.predict(df)
# print("Prediction:", prediction)

