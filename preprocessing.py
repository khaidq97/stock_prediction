
import numpy as np
import pandas as pd 
import os 
import datetime as dt
import urllib.request, json
# from sklearn.preprocessing import MinMaxScaler

# hàm tải dữ liệu
def load_data(ticker='AAL', data_source='alphavantage', number=200):
    if data_source == 'alphavantage':
      api_key = 'R1VDIP3TCVW9G57V'

      # ticker = 'AAL' # American Airlines stock market prices

      url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
      
      time = dt.datetime.now()
      
      file_to_save = 'stock_market_data-{}_{}.csv'.format(ticker,time.strftime("%Y-%m-%d"))
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # extract stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date','Open','High','Low','Close'])
            for k,v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(),float(v['1. open']),float(v['2. high']),
                              float(v['3. low']),float(v['4. close'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
        print('Data saved to : %s'%file_to_save)        
        df.to_csv(file_to_save)
        df = pd.read_csv(file_to_save, delimiter=',', usecols=['Date','Open','High','Low','Close'], index_col=0)
      # If the data is already there, just load it from the CSV
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save, delimiter=',', usecols=['Date','Open','High','Low','Close'], index_col=0)

    df = df.sort_values(by='Date',ascending=True)
    df['Mean'] = (df['Low'] + df['High'])/2
    data = df.iloc[:,4:].values

    return df[:number], data[:number]