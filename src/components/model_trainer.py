import os
import sys
from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from src.utils import save_object,evaluate_models

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.Model_trainer_config = ModelTrainerConfig()

    def initiat_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], # Every thing from np.array but not last column
                train_array[:,-1],  # Every rows but only last column
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Support Vector Classifier": SVC(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier()
            }
            params = {
                "Logistic Regression": {
                    "penalty": ["l2"],
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["lbfgs"],
                    "max_iter": [100, 200]
                },

                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                },

                "Decision Tree Classifier": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Random Forest Classifier": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                },

                "Support Vector Classifier": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"]
                },

                "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1]
                },

                "Gradient Boosting Classifier": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                    "subsample": [0.8, 1.0]
                }
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                models=models, params=params)
            
            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get best model nama
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(file_path=
                        self.Model_trainer_config.trained_model_file_path,
                        obj=best_model)
            
            prediction = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, prediction)

            return accuracy
        
        except Exception as e:
            raise CustomException(e, sys)
            