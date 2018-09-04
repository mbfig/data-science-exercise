import pandas as pd
import pickle
from utils_data import load_data
from utils_model import predict_a_week_ahead


def predict():
    path_data = '../dataset/usage_test.csv'
    path_model = './model/models.pickle'
    df = load_data(path_data)
    models = pickle.load(open(path_model, 'rb'))

    # do predictions
    predictions = pd.DataFrame(data=None)
    for house_id in df.id.unique():
        print('*********************')
        print('INFO: {}'.format(house_id))
        print('*********************')
        df_house = df[df.id == house_id]
        pred = predict_a_week_ahead(models[house_id], df_house)
        df_house = df_house.merge(pred[['ds', 'yhat']])
        predictions = predictions.append(df_house)

    save_predictions(predictions)


def save_predictions(df):
    df = df.rename(columns={'ds': 'datetime', 'yhat': 'usage'})
    df.to_csv('./results/usage_test_predictions.csv', index=False)


if __name__ == '__main__':
    predict()
