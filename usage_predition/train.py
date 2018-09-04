from utils_data import load_data
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import pickle
from utils_model import (
    fit_prophet_model,
    predict_a_week_ahead
)


def train():
    path = '../dataset/usage_train.csv'
    df = load_data(path)
    df['weekofyear'] = df.ds.apply(lambda x: x.weekofyear)
    samples_week = 2*24*7

    results_train = pd.DataFrame(data=None, index=df.id.unique(), columns=[50,60,70,80,90])
    results_test = results_train.copy()

    final_models = {}
    final_errors = {'train':{},'test':{} }
    for house_id in df.id.unique():
        df_house = df[df.id == house_id]
        print('*********************')
        print('INFO: {}'.format(house_id))
        print('*********************')
        results_train, results_test = error_estimation_cross_validation(
            df_house, house_id,
            samples_week, results_train, results_test
        )
        model, error_train, error_test = fit_final_model(df_house, samples_week)
        final_models[house_id] = model
        final_errors['train'][house_id] = error_train
        final_errors['test'][house_id] = error_test

    pickle.dump(final_models, open('./model/models.pickle', 'wb'))
    pickle.dump(final_errors, open('./model/final_errors.pickle', 'wb'))
    results_train.to_csv('./results/performance_metrics_train.csv')
    results_test.to_csv('./results/performance_metrics_test.csv')


def error_estimation_cross_validation(df_house, house_id,
    samples_week, expected_error_train, expected_error_test):

    # perform the outer-loop of a nested_cross validation 
    # for error estimation. we start by considering 50% 
    # of the weeks as training set and we increase by 10% until reach 90%

    # 
    # I didn't performed hyperparameters tuning, so no need 
    # for the inner loop with the training subset and the validation subset

    n_index = df_house.ds.nunique()
    for i in range(5, 10, 1):
        idx_split = math.floor(n_index * (i / 10))
        df_train_house, df_test_house = get_split(df_house, idx_split)

        model = fit_prophet_model(df_train_house)
    
        predict_training = predict_a_week_ahead(model, df_train_house[-samples_week:])
        expected_error_train.ix[house_id, (i * 10)] = calculate_NRMSE(
            df_train_house[-samples_week:],
            predict_training
        )

        forecast = predict_a_week_ahead(model, df_test_house)
        error = calculate_NRMSE(df_test_house, forecast)

        expected_error_test.ix[house_id, (i * 10)] = error

    return expected_error_train, expected_error_test


def fit_final_model(df_house, samples_week):
    n_index = df_house.ds.nunique()
    idx_split = math.floor(n_index * (0.99))
    df_train_house, df_test_house = get_split(df_house, idx_split)
    model = fit_prophet_model(df_train_house)

    predict_training = predict_a_week_ahead(model, df_train_house[-samples_week:])
    training_error = calculate_NRMSE(df_train_house[-samples_week:], predict_training)

    forecast = predict_a_week_ahead(model, df_test_house)
    test_error = calculate_NRMSE(df_test_house, forecast)

    return model, training_error, test_error


def get_split(df_house, idx_split):
    list_dates = list(df_house.ds.unique())
    threshold = list_dates[idx_split]

    df_train_house = df_house[df_house.ds <= threshold]
    df_test_house = df_house[
        (df_house.ds > threshold) &
        (df_house.ds <= threshold + np.timedelta64(7, 'D'))]
    
    return df_train_house, df_test_house


def calculate_NRMSE(true_values, forecast):
    results = true_values[['ds', 'y']].merge(forecast[['ds', 'yhat']], on='ds')
    max_y = true_values.y.max()
    min_y = true_values.y.min()
    return np.sqrt(mean_squared_error(results['y'], results['yhat']))/(max_y - min_y)



if __name__ == '__main__':
    train()