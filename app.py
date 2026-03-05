from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData
import pandas as pd

from sqlalchemy.orm import Session
from src.database import get_db,init_db
from src.schemas import PredictionCreate, PredictionUpdate
from src import crud

app = FastAPI()
templates = Jinja2Templates(directory="templates")
init_db()

# Hard coded.
# PCLASS_VALUES = [1, 2, 3]
# SEX_VALUES = ["male", "female"]
# EMBARKED_VALUES = ["C", "Q", "S"]


# Not hard code
df = pd.read_csv("artifacts/data.csv")
PCLASS_VALUES = list(df["pclass"].unique())
SEX_VALUES = list(df["sex"].unique())
# Having nan values
EMBARKED_NAN = df["embarked"].dropna()
EMBARKED_VALUES = list(EMBARKED_NAN.unique())


def base_context():
    return{
        "pclass_values": PCLASS_VALUES,
        "sex_values":SEX_VALUES,
        "embarked_values":EMBARKED_VALUES
    }


# Endpoint for the home page.
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")

# Register
@app.get("/register", response_class=HTMLResponse)
async def register_form(request:Request):
    return templates.TemplateResponse(
        request=request,name="register.html"
    )

@app.post("/register")
async def register(
                    user_id:str = Form(...),
                    name:str = Form(...),
                    db:Session = Depends(get_db)
                ):
    # Existing user -> show their prediction
    if crud.user_exists(db,user_id):
        return RedirectResponse(url=f"/predictions/{user_id}", status_code=303)
    
    # New user -> go to prediction form
    return RedirectResponse(url=f"/predict?user_id={user_id}&name={name}", status_code=303)


# Create
@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request,
                        user_id:str,
                        name:str):
    context = base_context()
    context.update({
        "user_id":user_id,
        "name":name
    })
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
                    request:Request,
                    db:Session = Depends(get_db),
                    user_id:str = Form(...),
                    name:str = Form(...),
                    pclass: int = Form(...),
                    sex: str = Form(...),
                    age: float = Form(...),
                    sibsp: int = Form(...),
                    parch: int = Form(...),
                    fare: float = Form(...),
                    embarked: str = Form(...)
                ):
    
    # Run model
    data = CustomData(pclass=pclass, sex=sex, age=age, sibsp=sibsp, parch=parch, fare=fare, embarked=embarked)
    features_df = data.get_data_as_data_frame()

    pipeline = PredictPipeline()
    prediction = pipeline.predict(features_df)[0]
    prediction_text = "Survived ✅" if prediction == 1 else "Did Not Survive ❌"

    # Save to database
    crud.create_prediction(db, PredictionCreate(user_id=user_id,name=name,pclass=pclass,
                                                sex=sex, age=age,sibsp=sibsp,parch=parch,fare=fare,
                                                embarked=embarked,prediction=prediction_text))
    context = base_context()
    context.update({
        "user_id":user_id,
        "name":name,
        "prediction_text":prediction_text
    })

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context
    )

# Read
@app.get("/predictions/{user_id}", response_class=HTMLResponse)
async def user_predictions(request:Request, user_id:str, db:Session=Depends(get_db)):
    records = crud.get_predictions_by_user(db=db, user_id=user_id)

    return templates.TemplateResponse(
        request=request,
        name="predictions.html",
        context={
            "records":records,
            "user_id":user_id
        }
    )

# Update
@app.get("/edit/{id}", response_class=HTMLResponse)
async def edit_form(request:Request, id:int, db:Session = Depends(get_db)):
    records = crud.get_prediction_by_id(db=db,id=id)
    context = base_context()
    context["record"] = records
    return templates.TemplateResponse(
        request=request,
        name="edit.html",
        context=context
    )

@app.post("/edit/{id}")
async def edit_prediction(
    id: int,
    db : Session = Depends(get_db),
    pclass: int = Form(...),
    sex: str = Form(...),
    age: float = Form(...),
    sibsp: int = Form(...),
    parch: int = Form(...),
    fare: float = Form(...),
    embarked: str = Form(...)
):
    records = crud.update_prediction(db,id,PredictionUpdate(
        pclass=pclass,
        sex=sex,
        age=age,
        sibsp=sibsp,
        parch=parch,
        fare=fare,
        embarked=embarked
    ))
    
    return RedirectResponse(url=f"/predictions/{records.user_id}", status_code=303)

# Delete
@app.post("/delete/{user_id}")
async def delete_user(user_id:str, db:Session=Depends(get_db)):
    crud.delete_user_predictions(db=db, user_id=user_id)
    return RedirectResponse(url="/", status_code=303)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)