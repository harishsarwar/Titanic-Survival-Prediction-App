import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scale = preprocessor.transform(features)

            preds = model.predict(data_scale)

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 pclass,
                 sex,
                 age,
                 sibsp,
                 parch,
                 fare,
                 embarked):
        self.pclass = pclass
        self.sex = sex
        self.age = age
        self.sibsp = sibsp
        self.parch = parch
        self.fare = fare
        self.embarked = embarked



    def get_data_as_data_frame(self):
        try:
            custome_data_input_dic = {
                    "pclass":[self.pclass],
                    "sex" : [self.sex],
                    "age" : [self.age],
                    "sibsp": [self.sibsp],
                    "parch" : [self.parch],
                    "fare" : [self.fare],
                    "embarked" : [self.embarked]

                }
                
            return pd.DataFrame(custome_data_input_dic)
        except Exception as e:
            raise CustomData(e,sys)
