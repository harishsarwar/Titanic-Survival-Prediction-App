from pydantic import BaseModel
from typing import Optional

class UserCreate(BaseModel):
    user_id:str
    name:str

class PredictionCreate(BaseModel):
    user_id:str
    name:str
    pclass:int
    sex:str
    age:float
    sibsp:int
    parch:int
    fare:float
    embarked:str
    prediction:str

class PredictionUpdate(BaseModel):
    pclass:   Optional[int]   = None
    sex:      Optional[str]   = None
    age:      Optional[float] = None
    sibsp:    Optional[int]   = None
    parch:    Optional[int]   = None
    fare:     Optional[float] = None
    embarked: Optional[str]   = None    