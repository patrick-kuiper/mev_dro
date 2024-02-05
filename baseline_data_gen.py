import math
import sys
import pickle

import torch
import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict 
import matplotlib.pyplot as plt




start_date = "1983-01-04"
end_date = "2023-12-31"
tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
# download and save data:
return_data = yf.download(tickers[tickers["Date added"] < start_date].Symbol.to_list(),
                   start_date, end_date, auto_adjust=True)['Close']
return_data.to_pickle("./return_data.p")

return_data = pd.read_pickle("./return_data.p")

sector_dict = {k:v for k, v in zip(tickers[tickers["Date added"] < start_date]["Symbol"], 
                                tickers[tickers["Date added"] < start_date]["GICS Sector"])}
return_data = pd.read_pickle("./return_data.p")
for ticker in return_data.keys():
    if (return_data[ticker].isnull() * 1).sum() > 2:
        print(ticker)
        return_data.drop(ticker, axis=1, inplace=True)
        del sector_dict[ticker]
return_data_start_date_filtered = return_data[return_data.index > start_date]
per_return_data_diff = (return_data_start_date_filtered.diff()[1:] / return_data_start_date_filtered[:-1])[:-1]


sector_dict_returns = defaultdict(list)
for co, ind in sector_dict.items():
    sector_dict_returns[ind].append(per_return_data_diff[co].to_numpy())
    
sector_dict_avg_returns = pd.DataFrame({ind: np.array(sector_dict_returns[ind]).mean(0) 
                                        for ind in sector_dict_returns.keys()})
sector_dict_avg_returns.index = per_return_data_diff.index

weekly_max = sector_dict_avg_returns[1:].resample('W').max()
montly_max = sector_dict_avg_returns[1:].resample('M').max()
yearly_max = sector_dict_avg_returns[1:].resample('Y').max()

weekly_max_return_data = torch.tensor(weekly_max.to_numpy())
monthly_max_return_data = torch.tensor(montly_max.to_numpy())
yearly_max_return_data = torch.tensor(yearly_max.to_numpy())
file = open('yearly_max_return_data.p', 'wb')
pickle.dump(yearly_max_return_data, file)
file.close()


weekly_max.plot(figsize=(24,8), alpha = 0.5)
plt.savefig("weekly_max_plot.png")

montly_max.plot(figsize=(24,8), alpha = 0.5)
plt.savefig("monthly_max_plot.png")

yearly_max.plot(figsize=(24,8), alpha = 0.5)
plt.savefig("yearly_max_plot.png")

file = open('monthly_max_return_data.p', 'wb')
pickle.dump(monthly_max_return_data, file)
file.close()

file = open('weekly_max_return_data.p', 'wb')
pickle.dump(weekly_max_return_data, file)
file.close()