

import pandas as pd

def load_data():

    df = pd.read_csv('/content/covid_19_clean_complete.csv',parse_dates=['Date'])

    df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

    return df

confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index


	

def monthly_cases(data):
    
    monthly_data = data.copy()

    # Drop the day indicator from the date column
    monthly_data.Date = monthly_data.date.apply(lambda x: str(x)[:-3])

    # Sum sales per month
    monthly_data = monthly_data.groupby('Date')['Confirmed'].sum().reset_index()
    monthly_data.Date = pd.to_datetime(monthly_data.Date)

    monthly_data.to_csv('../data/monthly_data.csv')

    return monthly_data
	

	

def get_diff(data):
    
    data['Confirmed_diff'] = data.Confirmed.diff()
    data = data.dropna()

    data.to_csv('../data/stationary_df.csv')

    return data


def generate_supervised(data):
    
    supervised_df = data.copy()

    #create column for each lag
    for i in range(1, 13):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['Confirmed_diff'].shift(i)

    #drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)

    supervised_df.to_csv('../data/model_df.csv', index=False)

def generate_arima_data(data):
    """Generates a csv file with a datetime index and a dependent sales column
    for ARIMA modeling.
    """
    dt_data = data.set_index('Date').drop('Confirmed', axis=1)
    dt_data.dropna(axis=0)

    dt_data.to_csv('../data/arima_df.csv')


def main():
    """Loads data from Kaggle, generates monthly dataframe and performs
    differencing to create stationarity. Exports csv files for regression
    modeling and for Arima modeling.
    """
    
    monthly_df = monthly_cases(confirmed)
    stationary_df = get_diff(monthly_df)

    generate_supervised(stationary_df)
    generate_arima_data(stationary_df)

main()
