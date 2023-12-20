# General functions for the Forecasting project
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from prettytable import PrettyTable
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
#---------------------- General functions ----------------------#

def select_country(df, country_name):
    """
    Selects data from a DataFrame for a specific country based on sensor IP.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing sensor data.
    - country_name (str): Name of the country for which data needs to be selected.

    Returns:
    - pd.DataFrame: Filtered DataFrame containing data only for the specified country.

    Example:
    >>> selected_data = select_country(df_sensor_data, 'United States')
    """

    # Find the available sensor IP of the country
    df_result = df[df['_source.hostGeoip.country_name'] == country_name]
    
    # Get the unique sensor IPs for the selected country
    available_IP = df_result['_source.hostIP'].value_counts().index.to_numpy()

    # Filter data using IP
    df_result = df[df['_source.hostIP'].isin(available_IP)]
    
    return df_result

def visualize_ts(df):
    """
    Visualize time series data by plotting the daily count of records over time.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing time series data.

    Returns:
    - pd.Series: Series containing the daily count of records.

    Example:
    >>> daily_records = visualize_ts(df_time_series_data)
    """

    # Convert 'startTime' to datetime
    df['date'] = pd.to_datetime(df['_source.startTime'])

    # Extract the date (day) from the timestamp
    df['date'] = pd.to_datetime(df['date'].dt.date)

    # Group by date and count the number of records
    daily_count = df.groupby('date').size()

    # Delete the last day because the request was performed that day in the morning,
    # i.e., not the whole day data available
    daily_count = daily_count[:-1]

    # Set up a stylish color palette
    colors = {'Daily Count': 'steelblue'}

    # Plot the time series
    plt.figure(figsize=(12, 8))
    daily_count.plot(linestyle='-', color=colors['Daily Count'])

    # Add labels
    plt.xlabel('Date')
    plt.ylabel('Cyberattacks')

    # Add legend and grid
    plt.grid(True)

    # Beautify the x-axis date labels
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    plt.show()

    return daily_count

def display_metrics_table(predictions, actual, countries):
    """
    Displays performance metrics (MAPE and RMSE) for predictions compared to actual values for each country.

    Parameters:
    - predictions (list of arrays): List of arrays containing predicted values for each country.
    - actual (list of arrays): List of arrays containing actual values for each country.
    - countries (list of str): List of country names corresponding to the predictions and actual values.

    Returns:
    - None: Prints a formatted table displaying metrics for each country.

    Example:
    >>> display_metrics_table(predictions, actual, ['United States', 'Canada', 'Germany'])
    +--------------+---------+--------+
    |   Country    | MAPE (%)|  RMSE  |
    +--------------+---------+--------+
    | United States|  5.23   | 12.45  |
    | Canada       |  8.12   | 18.76  |
    | Germany      |  6.45   | 15.32  |
    +--------------+---------+--------+
    """

    table = PrettyTable()
    table.field_names = ["Country", "MAPE (%)", "RMSE"]

    for country, preds, actuals in zip(countries, predictions, actual):
        mape = mean_absolute_percentage_error(actuals, preds) * 100
        rmse = np.sqrt(mean_squared_error(actuals, preds))

        table.add_row([country, f"{mape:.2f}", f"{rmse:.2f}"])

    print(table)





#---------------------- Classical methods ----------------------#

def trend(daily_count):
    """
    Visualize the trend component of a time series by plotting the original series
    and its rolling mean (simple moving average) with a window size of 7 days.

    Parameters:
    - daily_count (pd.Series): Series containing the daily count of records.

    Example:
    >>> trend_analysis = trend(daily_records)
    """

    # Calculate the rolling mean (simple moving average) with a window size of 7 days
    rolling_mean = daily_count.rolling(window=7).mean()

    # Set up a stylish color palette
    colors = {'Daily Count': 'steelblue'}

    # Visualize the original time series and the trend component
    plt.figure(figsize=(10, 6))
    plt.plot(daily_count.index, daily_count.values, label='Original', color=colors['Daily Count'])
    plt.plot(daily_count.index, rolling_mean, color='red', label='Trend (Moving Average)')
    plt.xlabel('Date')
    plt.ylabel('Cyberattacks')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

def plot_acf_pacf(country_name, country_data):
    """
    Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    for a given time series data of a specific country.

    Parameters:
    - country_name (str): Name of the country corresponding to the time series data.
    - country_data (pd.Series): Time series data for the specified country.

    Example:
    >>> plot_acf_pacf('United States', time_series_data_us)
    """

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 7))

    # Plot ACF
    plot_acf(country_data.values, ax=ax1)
    ax1.set_title(f'Autocorrelation Function (ACF) - {country_name}')

    # Plot PACF
    plot_pacf(country_data.values, ax=ax2)
    ax2.set_title(f'Partial Autocorrelation Function (PACF) - {country_name}')

    plt.tight_layout()
    plt.show()

def decomposition_ts(daily_count):
    """
    Perform advanced time series decomposition using Seasonal-Trend decomposition using LOESS (STL).

    Parameters:
    - daily_count (pd.Series): Time series data for decomposition.

    Example:
    >>> decomposition_ts(daily_records)
    """

    # Perform advanced decomposition using Seasonal-Trend decomposition using LOESS (STL)
    advanced_decomposition = STL(daily_count.values, period=7).fit()

    # Create subplots for observed, trend, seasonal, and residuals components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True)

    # Plot observed component
    ax1.plot(advanced_decomposition.observed)
    ax1.set_ylabel('Observed')

    # Plot trend component
    ax2.plot(advanced_decomposition.trend)
    ax2.set_ylabel('Trend')

    # Plot seasonal component
    ax3.plot(advanced_decomposition.seasonal)
    ax3.set_ylabel('Seasonal')

    # Plot residuals component
    ax4.plot(advanced_decomposition.resid)
    ax4.set_ylabel('Residuals')

    # Format x-axis for better readability
    fig.autofmt_xdate()

    # Set figure dimensions
    fig.set_figwidth(12)
    fig.set_figheight(8)

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()

def plot_results_ARIMA(pred, cf, train, test, country):
    """
    Plot the results of an ARIMA time series forecasting approach.

    Parameters:
    - pred (pd.Series): Predicted values.
    - cf (tuple): Confidence interval for the predictions.
    - train (pd.Series): Historic training data.
    - test (pd.Series): Actual test data.
    - country (str): Name of the country for which predictions were made.

    Example:
    >>> plot_results_ARIMA(predictions, confidence_interval, training_data, testing_data, 'United States')
    """

    # Align predicted values with test index
    pred = pd.Series(pred.values, index=test.index)

    # Create a subplot with two plots (2 rows, 1 column)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 14))

    # Plot the last month
    x = range(pred.size)
    axes[0].plot(x, test.values, label='Actual', linewidth=2.0, color='steelblue')
    axes[0].plot(x, pred, linewidth=2.0, color='orange', label='Predicted')
    axes[0].fill_between(x,
                         cf[0],
                         cf[1], color='grey', alpha=.3)
    
    # Plot beautiful x-axis
    date_objects = [datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S") for date in test.index]
    formatted_dates = [date.strftime("%Y-%m-%d") for date in date_objects]
    axes[0].set_xticks(range(0, len(formatted_dates), 5))
    axes[0].set_xticklabels(formatted_dates[::5], rotation=45, ha='right')
    
    # Calculate Mean Absolute Percentage Error (MAPE) for evaluation
    mape = mean_absolute_percentage_error(test.values, pred) * 100
    axes[0].set_title('Mean Absolute Percentage Error {0:.2f}%'.format(mape))
    axes[0].legend(loc='best')
    axes[0].grid(True)
    fig.show()

    # Plot the whole time series with teaching run and prediction
    if country == 'Spain':
        # For Spain --> ARIMA(1,0,1)
        model = ARIMA(train.values, order=(1, 0, 1))
    elif country == 'USA':
        # For USA --> ARIMA(0,1,2)
        model = ARIMA(train.values, order=(0, 1, 2))
    elif country == 'Singapore':
        # For Singapore --> ARIMA(3,0,1)
        model = ARIMA(train.values, order=(3, 0, 1))
    elif country == 'Germany':
        # For Germany--> ARIMA(1,0,2)
        model = ARIMA(train.values, order=(1, 0, 2))
    elif country == 'Japan':
        # For Japan--> ARIMA(1,0,4)
        model = ARIMA(train.values, order=(1, 0, 4))

    model_fit = model.fit()
    
    axes[1].plot(range(train.size), train.values, label='Historic', linewidth=2.0, color=(0.36, 0.73, 0.36))
    fitted_values = pd.Series(model_fit.fittedvalues, index=train.index)
    axes[1].plot(range(train.size), fitted_values, color='red', label='Teaching Run', linewidth=2.0)
    axes[1].plot(range(train.size, train.size + pred.size), pred, color='orange', label='Prediction', linewidth=2.0)
    axes[1].plot(range(train.size, train.size + pred.size), test.values, color='steelblue', label='Actual', linewidth=2.0)
    axes[1].fill_between(range(train.size, train.size + pred.size),
                         cf[0],
                         cf[1], color='grey', alpha=.3)
    axes[1].grid(True)

    plt.show()

def plot_results_prophet(pred, cf, train, test, country):
    """
    Plot the results of a prophet time series forecasting approach.

    Parameters:
    - pred (pd.Series): Predicted values.
    - cf (tuple): Confidence interval for the predictions.
    - train (pd.Series): Historic training data.
    - test (pd.Series): Actual test data.
    - country (str): Name of the country for which predictions were made.

    Example:
    >>> plot_results_prophet(predictions, confidence_interval, training_data, testing_data, 'United States')
    """

    # Align predicted values with test index
    pred = pd.Series(pred.values, index=test.index)

    # Create a subplot with two plots (2 rows, 1 column)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 14))

    # Plot the last month
    x = range(pred.size)
    axes[0].plot(x, test.values, label='Actual', linewidth=2.0, color='steelblue')
    axes[0].plot(x, pred, linewidth=2.0, color='orange', label='Predicted')
    axes[0].fill_between(x,
                         cf['yhat_lower'],
                         cf['yhat_upper'], color='grey', alpha=.3)
    
    # Plot beautiful x-axis
    date_objects = [datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S") for date in test.index]
    formatted_dates = [date.strftime("%Y-%m-%d") for date in date_objects]
    axes[0].set_xticks(range(0, len(formatted_dates), 5))
    axes[0].set_xticklabels(formatted_dates[::5], rotation=45, ha='right')
    
    # Calculate Mean Absolute Percentage Error (MAPE) for evaluation
    mape = mean_absolute_percentage_error(test.values, pred) * 100
    axes[0].set_title('Mean Absolute Percentage Error {0:.2f}%'.format(mape))
    axes[0].legend(loc='best')
    axes[0].grid(True)
    fig.show()

    # Plot the whole time series with teaching run and prediction
    if country == 'Spain':
        # For Spain --> ARIMA(1,0,1)
        model = ARIMA(train.values, order=(1, 0, 1))
    elif country == 'USA':
        # For USA --> ARIMA(0,1,2)
        model = ARIMA(train.values, order=(0, 1, 2))
    elif country == 'Singapore':
        # For Singapore --> ARIMA(3,0,1)
        model = ARIMA(train.values, order=(3, 0, 1))
    elif country == 'Germany':
        # For Germany--> ARIMA(1,0,2)
        model = ARIMA(train.values, order=(1, 0, 2))
    elif country == 'Japan':
        # For Japan--> ARIMA(1,0,4)
        model = ARIMA(train.values, order=(1, 0, 4))

    model_fit = model.fit()
    
    axes[1].plot(range(train.size), train.values, label='Historic', linewidth=2.0, color=(0.36, 0.73, 0.36))
    axes[1].plot(range(train.size, train.size + pred.size), pred, color='orange', label='Prediction', linewidth=2.0)
    axes[1].plot(range(train.size, train.size + pred.size), test.values, color='steelblue', label='Actual', linewidth=2.0)
    axes[1].fill_between(range(train.size, train.size + pred.size),
                         cf['yhat_lower'],
                         cf['yhat_upper'], color='grey', alpha=.3)
    axes[1].grid(True)
    axes[1].legend(loc='best')

    plt.show()

def prophet_data_format(df):
    """
    Format the input DataFrame for use with the Prophet time series forecasting model.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing time series data.

    Returns:
    - pd.DataFrame: Formatted DataFrame with columns 'ds' for dates and 'y' for values.

    Example:
    >>> formatted_df = prophet_data_format(time_series_data)
    """

    # Reset index and rename columns to 'ds' and 'y'
    df_prophet = df.reset_index()
    df_prophet.columns = ['ds', 'y']

    # Convert 'ds' column to datetime format
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    return df_prophet






#---------------------- Machine learning methods ----------------------#

def create_baseline_dataset(daily_count):
    """
    Creates a baseline dataset from daily count data, adding temporal features.

    Parameters:
    - daily_count (pd.Series): Daily count data with a datetime index.

    Returns:
    - pd.DataFrame: DataFrame with added temporal features for each day.

    Example:
    >>> baseline_df = create_baseline_dataset(daily_count)
    """

    df = pd.DataFrame({
        'count': daily_count,
        'month': daily_count.index.month,
        'year': daily_count.index.year,
        'dayofyear': daily_count.index.dayofyear,
        'weekofyear': daily_count.index.isocalendar().week,  # Use isocalendar to get ISO week
        'dayofweek': daily_count.index.dayofweek,
        'quarter': daily_count.index.quarter
    })

    # Create a binary feature 'workingday' based on the day of the week
    df['workingday'] = df['dayofweek'].apply(lambda x: 0 if x in [5, 6] else 1)

    return df

def train_val_test_split_ts(df):
    """
    Splits a time series DataFrame into training, validation, and test sets.

    Parameters:
    - df (pd.DataFrame): Time series DataFrame to be split.

    Returns:
    - tuple: Three DataFrames representing the training, validation, and test sets.

    Example:
    >>> train_df, val_df, test_df = train_val_test_split_ts(time_series_df)
    """

    # Last month for test
    test = df[-30:]

    # Previous month for validation
    val = df[-60:-30]

    # The rest for train
    train = df[:-60]

    return train, val, test

def add_lags(df, lag, label_col):
    """
    Adds lag features to a DataFrame based on the specified lag value.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - lag (int): Number of lag features to add.
    - label_col (str): Name of the target variable column.

    Returns:
    - pd.DataFrame: DataFrame with added lag features.

    Example:
    >>> lag_df = add_lags(input_df, 3, 'target_column')
    """

    # Copy the DataFrame to avoid modifying the original
    df_result = df.copy()

    # Add the lag of the target variable
    for i in range(1, lag + 1):
        df_result[f'lag_{i}'] = df_result[label_col].shift(i)
    
    # Drop rows with NaN values introduced by shifting
    df_result = df_result.dropna()

    return df_result
def create_rolling_features(df, columns, windows=[2, 3]):
    """
    Creates rolling features for specified columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list of str): List of column names for which rolling features are calculated.
    - windows (list of int, optional): List of window sizes for rolling calculations. Default is [2, 3].

    Returns:
    - pd.DataFrame: DataFrame with added rolling features.

    Example:
    >>> feature_df = create_rolling_features(input_df, ['column1', 'column2'], windows=[3, 6])
    """

    df_result = df.copy()

    for window in windows:
        df_result[f"rolling_mean_{window}"] = df_result[columns].shift(1).rolling(window=window).mean()
        df_result[f"rolling_std_{window}"] = df_result[columns].shift(1).rolling(window=window).std()
        df_result[f"rolling_var_{window}"] = df_result[columns].shift(1).rolling(window=window).var()
        df_result[f"rolling_min_{window}"] = df_result[columns].shift(1).rolling(window=window).min()
        df_result[f"rolling_max_{window}"] = df_result[columns].shift(1).rolling(window=window).max()
        df_result[f"rolling_min_max_ratio_{window}"] = df_result[f"rolling_min_{window}"] / df_result[f"rolling_max_{window}"]
        df_result[f"rolling_min_max_diff_{window}"] = df_result[f"rolling_max_{window}"] - df_result[f"rolling_min_{window}"]

    # Replace infinite values with NaN and fill NaN with 0
    df_result = df_result.replace([np.inf, -np.inf], np.nan)    
    df_result.fillna(0, inplace=True)

    return df_result








#---------------------- Deep learning methods ----------------------#