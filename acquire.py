import pandas as pd
import matplotlib.pyplot as plt

def get_seoul():
    '''
    Uses the GlobalLandTemperaturesByCountry.csv, 
    and GlobalLandTemperaturesByMajorCity.csv downloaded from 
    Kaggle (https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
    Returns the South Korea DataFrame, and Seoul DataFrame.
    '''
    df_country = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
    df_major_city = pd.read_csv("GlobalLandTemperaturesByMajorCity.csv")
    seoul = df_major_city[df_major_city.City == "Seoul"]
    south_korea = df_country[df_country.Country == "South Korea"]

    return south_korea, seoul


def clean_df(south_korea, seoul):
    '''
    Takes the south_korea, and seoul data frames,
    Runs data frames through the clean_sk(), and clean_seoul() functions
    Removes null columns, converts dt from object to datetime, set index to dt 
    drops irrelevent columns, renames columns for joining, and joins dataframes together.
    Plots the mean of seoul average temp and south_korea average temp at 1 month, 3 months, 6 months, and a year.
    Then adds a difference column, that takes the difference between the South Korean Average with the Seoul Average.
    Returns joined dataframe.
    '''
    south_korea = clean_sk(south_korea)
    seoul = clean_seoul(seoul)
    df = seoul.join(south_korea, how="left")
    df = df.tail(104)
    df.resample('M').mean().plot()
    df.resample('3M').mean().plot()
    df.resample('6M').mean().plot()
    df.resample('Y').mean().plot()
    plt.show()
    df['difference'] = df.south_korea_average_temp - df.seoul_average
    return df

def clean_seoul(seoul):
    '''
    Takes seoul dataframe
    Removes null columns, converts dt from object to datetime, set index to dt
    drops irrelevent columns, and renames columns for joining.
    returns seoul dataframe
    '''
    seoul = seoul[seoul.AverageTemperature.notnull()]
    seoul.dt = pd.to_datetime(seoul.dt)
    seoul = seoul.set_index('dt').sort_index()
    del seoul['City']
    del seoul['Country']
    del seoul['Latitude']
    del seoul['Longitude']
    del seoul['AverageTemperatureUncertainty']
    seoul = seoul.rename(columns={"AverageTemperature":"seoul_average"})
    return seoul

def clean_sk(south_korea):
    '''
    Takes south_korea dataframe
    Removes null columns, converts dt from object to datetime, set index to dt
    drops irrelevent columns, and renames columns for joining.
    returns south_korea dataframe.
    '''
    south_korea = south_korea[south_korea.AverageTemperature.notnull()]
    south_korea.dt = pd.to_datetime(south_korea.dt)
    south_korea = south_korea.set_index('dt').sort_index()
    del south_korea['AverageTemperatureUncertainty']
    del south_korea['Country']
    south_korea = south_korea.rename(columns={'AverageTemperature': 'south_korea_average_temp'})
    return south_korea

def wrangle_seoul():
    '''
    Uses get_seoul() and clean_df() to acquire the Seoul and South Korea Dataframes and clean them
    return the DataFrames joined as a single DataFrame.
    '''
    south_korea, seoul = get_seoul()
    df = clean_df(south_korea, seoul)
    return df