from fbprophet import Prophet


def fit_prophet_model(df_train_house):

    model = Prophet(interval_width=0.95)
    model.fit(df_train_house[['ds', 'y']])
    return model


def predict_a_week_ahead(model, future_dates):    
    forecast = model.predict(future_dates[['ds']])
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
