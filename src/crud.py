from sqlalchemy.orm import Session
from src.models import PredictionRecord
from src.schemas import PredictionCreate, PredictionUpdate


# Create
def create_prediction(db:Session, data:PredictionCreate):
    record = PredictionRecord(**data.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

# Read - check if user exists
def user_exists(db:Session, user_id:str):
    return db.query(PredictionRecord).filter(
        PredictionRecord.user_id == user_id
    ).first() is not None

# Read - get single record by id
def get_prediction_by_id(db:Session, id:int):
    return db.query(PredictionRecord).filter(
        PredictionRecord.id == id
    ).first()

# Read - get all record by the prediction by user
def get_predictions_by_user(db:Session, user_id:str):
    return db.query(PredictionRecord).filter(
        PredictionRecord.user_id == user_id
    ).all()


# Update
def update_prediction(db:Session, id:int, data:PredictionUpdate):
    record = db.query(PredictionRecord).filter(
        PredictionRecord.id == id
    ).first()

    if record:
        for key, value in data.model_dump(exclude_none=True).items():
            setattr(record, key, value)
        db.commit()
        db.refresh(record)
    return record


# Delete - delete all the predictions by user_id
def delete_user_predictions(db:Session, user_id:str):
    db.query(PredictionRecord).filter(
        PredictionRecord.user_id == user_id
    ).delete()
    db.commit()