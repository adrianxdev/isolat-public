#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import sklearn as skl
import scipy.stats as stats
import sys
local_rel_path = '../data/'
sys.path.insert(0, local_rel_path)
import nytimes
import importlib
import models
import features
importlib.reload(models)
importlib.reload(features)

import os



import kinsa
import pandas as pd
from shutil import copyfile


# In[2]:


# get the data in

state_df, county_df = nytimes.get_nyt_data()

county_cases_ts = nytimes.convert_county_df_to_ts(county_df, quantity='cases')
county_deaths_ts = nytimes.convert_county_df_to_ts(county_df, quantity='deaths')
county_fips = nytimes.convert_county_df_to_ts(county_df, quantity='fips')

state_cases_ts = nytimes.convert_state_df_to_ts(state_df, quantity='cases')
state_deaths_ts = nytimes.convert_state_df_to_ts(state_df, quantity='deaths')

def preprocess_df(df):
    
    # drop county, state columns if exist
    df = df.drop(columns=['county', 'state'], errors='ignore')
    # fill nas with zeros
    df = df.fillna(0)
    # replace column indices with datetime objects
    df = df.rename(
        columns=lambda str_date: dt.datetime.strptime(str_date, '%m/%d/%y'))
    
    return df

# do standard preprocessing below:
state_cases_ts = preprocess_df(state_cases_ts)
state_deaths_ts = preprocess_df(state_deaths_ts)

county_cases_ts = preprocess_df(county_cases_ts)
county_deaths_ts = preprocess_df(county_deaths_ts)


# In[50]:


# demonstrate base forecast for Middlesex, Massachusetts


demo_county = ('Massachusetts', 'Middlesex')

#demo_county = ('California', 'Kern')

out = ['date','value']

demo_ts = county_cases_ts.loc[demo_county , :].copy()
demo_ts_daily = demo_ts.diff()
# since we are demonstrating a base, not rolling forecast, shorten the 
# time series to the relevant portion, i.e. where case count is large
demo_ts_daily_short = demo_ts_daily.loc[demo_ts > 10]
# now implement data checks, i.e. make sure daily data is positive
demo_ts_daily_short = demo_ts_daily_short*(demo_ts_daily_short>=0)

# now compute the forecasts
target_date_range = pd.date_range(demo_ts_daily_short.index[0], 
                                  demo_ts_daily_short.index[-1]+dt.timedelta(days=21))
out = models.base_forecast_linear(demo_ts_daily_short, target_date_range=target_date_range) #point
out80 = models.base_forecast_linear(demo_ts_daily_short, quantile=0.8, target_date_range=target_date_range) # upper quantile
out20 = models.base_forecast_linear(demo_ts_daily_short, quantile=0.2, target_date_range=target_date_range) # lower quantile


out.head(10)

df = pd.DataFrame(data = out )
df['date'] = df.index
df['value'] = df[0]
df_final = df[['date','value']].copy()
df_final.to_csv("mass_middle_auto.csv", index=False)

#copyfile("mass_middle_auto.csv", r"C:\Users\adrian\IDSS-hack\mass_middlesex_auto.csv")

