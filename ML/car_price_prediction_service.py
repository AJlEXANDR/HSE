from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import io
import nest_asyncio
import uvicorn

nest_asyncio.apply()

app = FastAPI()

class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    torque: float
    seats: int
    max_torque_rpm: float


class Items(BaseModel):
    objects: List[Item]


def train_model():
    train_url = 'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv'
    df_train = pd.read_csv(train_url)

    df_train = df_train.drop_duplicates(subset=df_train.columns.difference(['selling_price']), keep='first')

    def preprocess_columns(data):
        data['mileage'] = data['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
        data['engine'] = data['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
        data['max_power'] = data['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)

        torque_split = data['torque'].str.extract(r'(\d+\.?\d*)Nm(?:@ ?(\d+))?')
        data['torque'] = torque_split[0].astype(float)
        data['max_torque_rpm'] = torque_split[1].astype(float)

        return data

    df_train = preprocess_columns(df_train)

    for col in ['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']:
        df_train[col] = df_train[col].fillna(df_train[col].median())

    df_train['engine'] = df_train['engine'].astype(int)
    df_train['seats'] = df_train['seats'].astype(int)

    y_train = df_train['selling_price']
    X_train = df_train.drop(columns=['selling_price', 'name'])

    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(exclude=['object'])
    X_train[num_cols.columns] = scaler.fit_transform(num_cols)

    X_train = pd.get_dummies(X_train, dtype='int', drop_first=True)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, scaler


model, scaler = train_model()


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    try:
        data = pd.DataFrame([item.dict()])
        num_cols = data.select_dtypes(exclude=['object'])

        data[num_cols.columns] = scaler.transform(num_cols)

        data = pd.get_dummies(data, dtype='int', drop_first=True)

        missing_cols = set(model.feature_names_in_) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        data = data[model.feature_names_in_]

        prediction = model.predict(data)[0]
        return round(prediction, 2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_items")
def predict_items(file: UploadFile):
    try:
        content = file.file.read()
        df = pd.read_csv(io.BytesIO(content))

        num_cols = df.select_dtypes(exclude=['object'])
        df[num_cols.columns] = scaler.transform(num_cols)
        df = pd.get_dummies(df, dtype='int', drop_first=True)

        missing_cols = set(model.feature_names_in_) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[model.feature_names_in_]

        predictions = model.predict(df)
        df['predicted_price'] = predictions

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        response = StreamingResponse(
            output,
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def run_app():
    uvicorn.run(app, host="127.0.0.1", port=8001)
