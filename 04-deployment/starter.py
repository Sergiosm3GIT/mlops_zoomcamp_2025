

import pickle
import pandas as pd
import numpy as np
import os
import sys

categorical = ['PULocationID', 'DOLocationID']

def read_data(color, year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(input_file)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df


def load_model(filename):
    with open(filename, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def apply_model(color, year, month, model_file):
    df = read_data(color, year, month)
    dv, model = load_model(model_file)
    output_file = f'output/{color}/{year:04d}-{month:02d}.parquet'
    output_dir = os.path.dirname(output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    df['prediction'] = y_pred 

    df_result = df[['ride_id', 'prediction']].copy()
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return df_result

def run():
    color = sys.argv[1] # 'yellow'
    year = int(sys.argv[2]) # 2023
    month = int(sys.argv[3]) # 3
    filename = 'model.bin'
    
    df_results = apply_model(color, year, month, filename)
    print(df_results.prediction.mean())

if __name__ == '__main__':
    run()
    print('Done!')


