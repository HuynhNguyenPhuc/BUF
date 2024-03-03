from utils.loss import mse, rmse, mape, r2_score
from utils.getter import getStockExchangeList, getTrainingData, getPretrainedModel

import pandas as pd

def evaluation():
    stock_exchanges = getStockExchangeList()

    input_data, label_data = getTrainingData()
    model = getPretrainedModel()

    mse_ = []
    rmse_ = []
    mape_ = []
    r2_score_ = []

    for exchange in stock_exchanges:
        y_true = label_data[exchange]
        y_pred = model[exchange].predict(input_data[exchange], verbose = 0)

        mse_.append(mse(y_true, y_pred).numpy())
        rmse_.append(rmse(y_true, y_pred).numpy())
        mape_.append(mape(y_true, y_pred).numpy())
        r2_score_.append(r2_score(y_true, y_pred).numpy())

    results_df = pd.DataFrame({
        'Exchange': stock_exchanges,
        'MSE': mse_,
        'RMSE': rmse_,
        'MAPE': mape_,
        'R2 Score': r2_score_
    })

    # Save the result of model evaluation to a csv file
    results_df.to_csv('./result/result.csv', index = False) 
    print(results_df)