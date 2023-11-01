from enum import Enum

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy import create_engine


class Base(DeclarativeBase):
    pass


class Iris(Base):
    __tablename__ = "iris"

    iris_id: Mapped[int] = mapped_column(primary_key=True)
    petal_height: Mapped[float]
    petal_width: Mapped[float]
    sepal_height: Mapped[float]
    sepal_width: Mapped[float]


app = FastAPI()


class IrisType(str, Enum):
    SETOSA = "setosa"
    VERSICOLOUR = "versicolour"
    VIRGINICA = "virginica"


class IrisModel(BaseModel):
    petal_height: float = Field(ge=0)
    petal_width: float = Field(ge=0)
    sepal_height: float = Field(ge=0)
    sepal_width: float = Field(ge=0)


@app.get("/healthcheck")
def healthcheck():
    return {"status": "OK"}


def fetch_data() -> pd.DataFrame:
    engine = create_engine(url="postgresql+psycopg2://admin:admin@localhost:5432/iris")

    with Session(engine) as session:
        query = session.query(Iris).filter(Iris.petal_width > 10)
        return pd.read_sql(query.statement, query.session.bind).set_index("iris_id")


def get_predict(data: pd.DataFrame, iris: IrisModel) -> int:
    model = KNeighborsClassifier()
    model.fit(X=data[["petal_height", "petal_width", "sepal_height", "sepal_width"]], y=data["target"])
    return model.predict(pd.DataFrame(data=iris.model_dump(), index=[0]).values)[0]


@app.get("/predict")
def predict(iris: IrisModel):
    data = fetch_data()
    predict = get_predict(data=data, iris=iris)

    iris_type = {
        0: IrisType.SETOSA,
        1: IrisType.VERSICOLOUR,
        2: IrisType.VIRGINICA,
    }[predict]
    return {"predict": iris_type}


if __name__ == "__main__":
    fetch_data()
