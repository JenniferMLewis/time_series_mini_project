import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


def split(df):
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    validate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]
    for col in train.columns:
        plt.figure(figsize=(12,4))
        plt.plot(train[col])
        plt.plot(validate[col])
        plt.plot(test[col])
        plt.ylabel(col)
        plt.show()
    return train, validate, test

def histoplots(df):
    sns.histplot(data= df, x='seoul_average',color = 'orange')
    plt.show()
    sns.histplot(data= df, x='south_korea_average_temp')
    plt.show()
    sns.histplot(data= df, x='seoul_average', color = 'orange')
    sns.histplot(data= df, x='south_korea_average_temp')
    plt.xlabel('Temperatures')
    plt.show()

def correlation(df):
    '''
    Takes in a dataframe
    Returns the corr, p from stats.pearsonr for seoul_average, and south_korea_average_temp
    '''
    corr, p = stats.pearsonr(df.seoul_average, df.south_korea_average_temp)
    print (f'''Correlation : {corr}
    p-value : {p}''')
    df.plot.scatter('seoul_average', 'south_korea_average_temp')
    plt.title("Seoul's Average Temp vs. South Korea's Average Temp")
    plt.text(0, 20, f'corr = {corr:.3f}')
    plt.show()

    plt.plot(df.seoul_average, df.south_korea_average_temp)
    plt.title('Seoul Temp vs South Korea Temp')
    plt.xlabel('Seoul Temperatures')
    plt.ylabel('South Korea Temperatures')

    plt.text(-3, 17, f'corr = {corr:.2f}')
    plt.text(-3, 15, f'p = {p:e}')
    plt.show()
    return corr, p

def create_y(train):
    y = train.seoul_average
    y2 = train.south_korea_average_temp
    return y, y2

def decomp(y, y2):
    result = sm.tsa.seasonal_decompose(y)
    decomposition = pd.DataFrame({
        'y': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid,
    })
    print("Seoul Seasonality Decomposition:")
    print(decomposition.head())
    print("-------")
    result2 = sm.tsa.seasonal_decompose(y2)
    decomposition2 = pd.DataFrame({
        'y': result2.observed,
        'trend': result2.trend,
        'seasonal': result2.seasonal,
        'resid': result2.resid,
    })
    print("South Korea Seasonality Decomposition:")
    print(decomposition2.head())

    decomposition.iloc[:, 1:].plot()
    decomposition.plot()
    result.plot()
    plt.show()

    decomposition2.iloc[:, 1:].plot()
    decomposition2.plot()
    result2.plot()
    plt.show()
    return decomposition, decomposition2

def plot_y(y, y2):
    ax = y.groupby(y.index.month).mean().plot.bar(width=.9, ec='black')
    plt.xticks(rotation=0)
    ax.set(title='Average Temperature For Seoul by Month', xlabel='Month', ylabel='Temp (C)')
    plt.show()
    ax = y.groupby(y.index.year).mean().plot.bar(width=.9, ec='black')
    plt.xticks(rotation=0)
    ax.set(title='Average Temperature For Seoul by Year', xlabel='Year', ylabel='Temp (C)')
    plt.show()
    ax = y2.groupby(y2.index.month).mean().plot.bar(width=.9, ec='black')
    plt.xticks(rotation=0)
    ax.set(title='Average Temperature For South Korea by Month', xlabel='Month', ylabel='Temp (C)')
    plt.show()
    ax = y2.groupby(y2.index.year).mean().plot.bar(width=.9, ec='black')
    plt.xticks(rotation=0)
    ax.set(title='Average Temperature For South Korea by Year', xlabel='Month', ylabel='Temp (C)')
    plt.show()
    

def y_tests(train):
    y, y2 = create_y(train)
    plot_y(y, y2)
    decomp(y, y2)


α = 0.05
def yearly_var(train):
    oh_five = train.seoul_average[train.index.year == 2005]
    oh_six = train.seoul_average[train.index.year == 2006]
    oh_seven = train.seoul_average[train.index.year == 2007]
    oh_eight = train.seoul_average[train.index.year == 2008]
    stat, p = stats.bartlett(oh_five, oh_six, oh_seven, oh_eight)
    if p < α:
        print("We fail to reject a Null hypothesis. There is not sufficient evidence to say the groups have different variances.")
    else:
        print("We can reject the Null Hypothesis. The groups appear to have different variances")


def ttest_five_six(train):
    oh_five = train.seoul_average[train.index.year == 2005]
    oh_six = train.seoul_average[train.index.year == 2006]
    t, p = stats.ttest_ind(oh_five, oh_six, equal_var=False)
    if p < α:
        print("We can reject the Null Hypothosis. There is some significant difference.")
    else:
        print("We cannot reject the Null Hypothosis, there is little to no significant difference.")

def ttest_six_seven(train):
    oh_six = train.seoul_average[train.index.year == 2006]
    oh_seven = train.seoul_average[train.index.year == 2007]
    t, p = stats.ttest_ind(oh_six, oh_seven, equal_var=False)
    if p < α:
        print("We can reject the Null Hypothosis. There is some significant difference.")
    else:
        print("We cannot reject the Null Hypothosis, there is little to no significant difference.")

def ttest_seven_eight(train):
    oh_seven = train.seoul_average[train.index.year == 2007]
    oh_eight = train.seoul_average[train.index.year == 2008]
    t, p = stats.ttest_ind(oh_seven, oh_eight, equal_var=False)
    if p < α:
        print("We can reject the Null Hypothosis. There is some significant difference.")
    else:
        print("We cannot reject the Null Hypothosis, there is little to no significant difference.")