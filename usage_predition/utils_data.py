import pandas as pd


def load_data(path):
    df = pd.read_csv(path, date_parser=pd.to_datetime)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by='datetime')
    # rename_columns for prophet
    df = df.rename(columns={'datetime': 'ds', 'usage': 'y'})

    return df