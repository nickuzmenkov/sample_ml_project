import pandas as pd
import psycopg2
from sklearn.datasets import load_iris


def create_table() -> None:
    query = """
    CREATE TABLE IF NOT EXISTS Iris (
        iris_id SERIAL PRIMARY KEY,
        petal_height FLOAT,
        petal_width FLOAT,
        sepal_height FLOAT,
        sepal_width FLOAT,
        target INT
    )
    """

    with psycopg2.connect(
        dbname='iris',
        user='admin',
        password='admin',
        host='0.0.0.0'
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            connection.commit()


def fill_iris_data() -> None:
    query = """
        INSERT INTO Iris (iris_id, petal_height, petal_width, sepal_height, sepal_width, target)
        VALUES ({iris_id}, {petal_height}, {petal_width}, {sepal_height}, {sepal_width}, {target})
        ON CONFLICT DO NOTHING
    """
    data = load_iris()
    df = pd.DataFrame(
        data=data["data"],
        columns=[
            "petal_height",
            "petal_width",
            "sepal_height",
            "sepal_width",
        ],
    )
    df["target"] = data["target"]

    with psycopg2.connect(
        dbname='iris',
        user='admin',
        password='admin',
        host='localhost'
    ) as connection:
        with connection.cursor() as cursor:
            for index, row in df.iterrows():
                cursor.execute(
                    query.format(
                        iris_id=index,
                        petal_height=row["petal_height"],
                        petal_width=row["petal_width"],
                        sepal_height=row["sepal_height"],
                        sepal_width=row["sepal_width"],
                        target=row["target"],
                    )
                )
            connection.commit()


def fetch_data() -> None:
    query = """
    SELECT iris_id, petal_height, petal_width, sepal_height, sepal_width, target
    FROM Iris
    """

    with psycopg2.connect(
        dbname='iris',
        user='admin',
        password='admin',
        host='0.0.0.0'
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            df = pd.DataFrame(
                rows,
                columns=[
                    "iris_id",
                    "petal_height",
                    "petal_width",
                    "sepal_height",
                    "sepal_width",
                    "target",
                ],
            ).set_index("iris_id")
            print(df)


if __name__ == "__main__":
    create_table()
    fill_iris_data()
    fetch_data()
