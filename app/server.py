from typing import List

import predict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

templates = Jinja2Templates(directory="templates")


app = FastAPI()


# Модели для типизированного вывода
class BookModel(BaseModel):
    id: str
    title: str
    author: str


class RecommendationsModel(BaseModel):
    recommendations: List[BookModel]
    history: List[BookModel]


# Инициализируем на уровне модуля, чтобы данные не загружались на каждый запрос
predictor = predict.Predictor()


@app.get("/recommendations/{user_id}", response_model=RecommendationsModel)
async def get_recommendations(user_id: int):
    history = predictor.get_history(user_id)
    user_age = predictor.get_age(user_id)

    recommendations = predictor.recommend(history=history, user_age=user_age)

    return RecommendationsModel(
        recommendations=[
            BookModel(id=str(row.recId), title=str(row.title), author=(row.author_fullName))
            for _, row
            in recommendations.iterrows()
        ],
        history=[
            BookModel(id=str(row.recId), title=str(row.title), author=str(row.author_fullName))
            for _, row
            in history.iterrows()
        ]
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
