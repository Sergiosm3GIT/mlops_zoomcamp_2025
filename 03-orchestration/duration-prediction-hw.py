#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from prefect import flow, task

import datetime
from dateutil.relativedelta import relativedelta

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


@task(retries=3,retry_delay_seconds=2,log_prints=True)
def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    print(f"Reading data from {url}")
    df = pd.read_parquet(url)
    print(f"Data registers from {year}-{month:02d} : {df.shape}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    print(f"Data shape after filtering by duration from {year}-{month:02d} : {df.shape}")
    return df

@task
def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

@task
def train_model(X_train, y_train, dv):
    with mlflow.start_run() as run:
        mlflow.set_tag("yellow taxi", "duration-prediction")
        lr = LinearRegression().fit(X_train, y_train)
        print(f"Model intercept: {lr.intercept_}")
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")
        return run.info.run_id

@task(log_prints=True)
def run(year, month):
    df_train = read_dataframe(year=year, month=month)
    X_train, dv = create_X(df_train)
    target = 'duration'
    y_train = df_train[target].values

    run_id = train_model(X_train, y_train,dv)
    print(f"MLflow run_id: {run_id}")
    return run_id


@flow(log_prints=True)
def main_flow(year: int, month: int):
    # Call the task with parameters
    run_id = run(year=year, month=month)

    # Save the run_id to a file
    with open("run_id.txt", "w") as f:
        f.write(run_id)

if __name__ == "__main__":
    # Default parameters 
    #train_date =  datetime.date.today().replace(day=1) - relativedelta(months=4) 
    year = 2023 #train_date.year
    month = 3 #train_date.month
    main_flow(year = year, month = month)