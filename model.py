import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import Holt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from explore import split


def create_yhat(train, validate):
    '''
    Takes in Train and Validate
    creates a yhat using seoul_average
    returns yhat_df
    '''
    seoul_temp = train['seoul_average'][-1:][0]
    yhat_df = pd.DataFrame({'seoul_temp': [seoul_temp]}, 
                       index = validate.index)
    return yhat_df

def holts_model(train, validate, yhat_df, eval_df):
    '''
    Takes in train, validate, yhat_df, and eval_df
    Trains to the Holt Prediction Model
    Adds predictions to yhat
    Uses plot_and_eval() to plot out and record evaluations on eval_df
    Returns yhat_df and eval_df
    '''
    for col in train.columns:
        model = Holt(train[col], exponential = False)
        model = model.fit(smoothing_level = .1, 
                        smoothing_slope = .1, 
                        optimized = False)
        yhat_temp = model.predict(start = validate.index[0], 
                                end = validate.index[-1])
        yhat_df[col] = round(yhat_temp, 2)
    for col in train.columns:
        eval_df = append_eval_df(validate, yhat_df, eval_df, model_type = 'Holts', 
                                 target_var = col)
    for col in train.columns:
        plot_and_eval(train, validate, yhat_df, target_var = col)
    print(eval_df)
    return yhat_df, eval_df

def prev_cycle(df, eval_df):
    '''
    Takes in DataFrame and eval_df
    Creates New train, validate, and test that split at a yearly cycle
    Creates predictions using Previous Year.
    Returns new train, validate, test, and yhat_df, and eval_df
    '''
    train_pre = df[:'2011-08']
    validate_pre = df["2011-09" :"2012-08"]
    test_pre = df['2012-09': '2013-08']
    yhat_df = train_pre['2010-09':'2011-08'] + train_pre.diff(12).mean()
    yhat_df.index = validate_pre.index
    for col in train_pre.columns:
        plot_and_eval(train_pre, validate_pre, yhat_df, target_var = col)
        eval_df = append_eval_df(validate_pre, yhat_df, eval_df, model_type = 'previous year', target_var = col)
    print(eval_df)
    return train_pre, validate_pre, test_pre, yhat_df, eval_df


def fb_prophet(df):
    '''
    Dates in a DataFrame
    Creates New Dataframe using the Data to the specifications in Prophet's documentation,
    Splits the New Dataframe to Train, Validate, and Test
    Fits to Prophet, Creates a Future Dataframe,
    Predicts the Future
    Prints Results and Plots the forecast the model created.
    Returns train, validate, test
    '''
    seoul_temps = df.copy()
    del seoul_temps['south_korea_average_temp']
    del seoul_temps['difference']
    seoul_temps = seoul_temps.reset_index()
    seoul_temps = seoul_temps.rename(columns={'dt':'ds', 'seoul_average': 'y'})
    train_p, validate_p, test_p = split(seoul_temps)
    m = Prophet()
    m.fit(train_p)
    future = m.make_future_dataframe(periods=1)
    future = future[:-1]
    future.tail()
    f = {'ds': ['2009-05-01', '2009-06-01', '2009-07-01', '2009-08-01', '2009-09-01', '2009-10-01', '2009-11-01', '2009-12-01', '2010-01-01', '2010-02-01', '2010-03-01', '2010-04-01']}
    f = pd.DataFrame(f)
    future = future.append(f, ignore_index = True)
    future.ds = pd.to_datetime(future.ds)
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15))
    m.plot(forecast)
    plot_plotly(m, forecast)
    plot_components_plotly(m, forecast)
    return train_p, validate_p, test_p

def best_rmse(eval_df, train):
    '''
    Takes in eval_df and train
    finds the lowest rmse,
    prints the lowest rmse as a table
    Bar Plots the comparisions.
    '''
    # get the min rmse for each variable

    min_rmse_seoul = eval_df.groupby('target_var')['rmse'].min()[0]
    min_rmse_sk = eval_df.groupby('target_var')['rmse'].min()[1]

    # filter only the rows that match those rmse to find out 
    # which models are best thus far
    print(eval_df[((eval_df.rmse == min_rmse_seoul) | 
            (eval_df.rmse == min_rmse_sk)
            )])

    for col in train.columns:
        x = eval_df[eval_df.target_var == col]['model_type']
        y = eval_df[eval_df.target_var == col]['rmse']
        plt.figure(figsize=(12, 6))
        sns.barplot(x, y)
        plt.title(col)
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.show()


def test_model(train_pre, validate_pre, test_pre):
    '''
    Takes in Previous Year's train, validate, and test
    Evaluates the Previous Year model on Test
    Prints the Results
    and Plots them.
    '''
    yhat_df = validate_pre + train_pre.diff(12).mean()
    yhat_df.index = test_pre.index
    rmse_temp = round(sqrt(mean_squared_error(test_pre['seoul_average'], yhat_df['seoul_average'])), 2)
    rmse_sk_average_temp = round(sqrt(mean_squared_error(test_pre['south_korea_average_temp'], yhat_df['south_korea_average_temp'])), 2)
    rmse_difference = round(sqrt(mean_squared_error(test_pre['difference'], yhat_df['difference'])), 2)
    print("On Test:")
    print(f"rmse - seoul_average: {rmse_temp}")
    print(f"rmse - sk_average_temp: {rmse_sk_average_temp}")
    print(f"rmse - difference: {rmse_difference}")
    for col in train_pre.columns:
        plot_and_eval_test(train_pre, validate_pre, test_pre, yhat_df, col)



def plot_and_eval_test(train, validate, test, yhat_df, target_var):
    '''
    Takes train, validate, test, yhat_df, and target_var
    Plots and Evaluates the results of Evaluating on Test.
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(test[target_var], label = 'Test', linewidth = 1)
    plt.plot(yhat_df[target_var], alpha = .5, color="red")
    plt.title(target_var)
    plt.legend()
    plt.show()






def evaluate(validate, yhat_df, target_var):
    '''
    Takes in validate, yhat_df, and target_var
    Uses RMSE to evaluate Validate vs the yhat prediction.
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse

def plot_and_eval(train, validate, yhat_df ,target_var):
    '''
    Takes Train, Validate, yhat_df, and target_var
    Plots the results of predictions, and Prints the resulting RMSE
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(validate, yhat_df, target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()

# Create the empty dataframe
def create_eval():
    '''
    Creates a blank eval_df dataframe with columns model_type, target_var, and rmse
    returns the dataframe
    '''
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    return eval_df

# function to store rmse for comparison purposes
def append_eval_df(validate, yhat_df, eval_df, model_type, target_var):
    '''
    Takes in validate, yhat_df, eval_df, model_type, and target_var
    Evalutes the predictions using RMSE
    Creates data to append to the eval_df dataframe.
    Returns the data appended to the eval_df
    '''
    rmse = evaluate(validate, yhat_df, target_var)
    d = {'model_type': [model_type], 'target_var': [target_var], 'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)