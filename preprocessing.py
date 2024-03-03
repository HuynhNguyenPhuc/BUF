from utils.getter import getStockExchangeList

import pandas as pd
import os

THRESHOLD = 1000

def preprocessing():
    stock_exchanges = getStockExchangeList()
    df = {i:{} for i in stock_exchanges}

    for exchange in stock_exchanges:
        exchange_directory = f"./data/raw_data/{exchange}"
        for file in sorted(os.listdir(exchange_directory)):
            stock_name = file[:-4]
            df_key = stock_name[:3] if len(stock_name) > 3 else stock_name
            df[exchange][df_key] = pd.concat([df[exchange].get(df_key, pd.DataFrame()), pd.read_csv(f"{exchange_directory}/{file}")])

    for exchange in list(df.keys()):
        try:
            os.mkdir(f'./data/processed_data/{exchange}')
        except:
            pass
        for stock in list(df[exchange].keys()):
            dataframe = df[exchange][stock]
            dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%m/%d/%Y')
            dataframe = dataframe.sort_values(by=['Date'])
            last_date_year = dataframe['Date'].dt.year.values.tolist()[-1]
            if dataframe.shape[0] < THRESHOLD or last_date_year != 2024:
                del df[exchange][stock]
            else:
                dataframe['Date'] = dataframe['Date'].dt.strftime('%m/%d/%Y')
                df[exchange][stock] = dataframe.tail(750).reset_index()
                df[exchange][stock].to_csv(f'./data/processed_data/{exchange}/{stock}.csv')