from sqlalchemy import Column, Integer, String, Float
from src.database import Base



class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable= False)
    name = Column(String, nullable=False)
    pclass = Column(Integer)
    sex = Column(String)
    age = Column(Integer)
    sibsp = Column(Integer)
    parch = Column(Integer)
    fare = Column(Float)
    embarked = Column(String)
    prediction = Column(String)