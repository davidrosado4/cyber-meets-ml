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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.decomposition import PCA
import seaborn as sns
import optuna
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

def select_continent(df, continent_name):
    """
    Selects data from a DataFrame for a specific continent based on sensor IP.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing sensor data.
    - continent_name (str): Name of the continent for which data needs to be selected.

    Returns:
    - pd.DataFrame: Filtered DataFrame containing data only for the specified continent.

    Example:
    >>> selected_data = select_continent(df_sensor_data, 'EU')
    """

    # Find the available sensor IP of the country
    df_result = df[df['_source.hostGeoip.continent_code'] == continent_name]
    
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

def objective(trial, model_type, X_train, X_val, y_train, y_val):
    """
    Objective function for hyperparameter optimization using Optuna.

    Parameters:
    - trial (optuna.Trial): Optuna trial object for optimization.
    - model_type (str): Type of model for optimization (e.g., 'RandomForest', 'GradientBoosting', 'XGBoost').
    - X_train (pd.DataFrame): Training features.
    - X_val (pd.DataFrame): Validation features.
    - y_train (pd.Series): Training target variable.
    - y_val (pd.Series): Validation target variable.

    Returns:
    - float: Mean squared error of the model predictions on the validation set.

    Example:
    >>> mse = objective(trial, 'RandomForest', X_train, X_val, y_train, y_val)
    """

    if model_type == 'RandomForest':
        # Define search for hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
            'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        model = RandomForestRegressor(random_state=42, **params)

    elif model_type == 'GradientBoosting':
        # Define the search for hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        model = GradientBoostingRegressor(random_state=42, **params)

    elif model_type == 'XGBoost':
        # Define the search for hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'alpha': trial.suggest_float('alpha', 0.0, 1.0),
            'lambda': trial.suggest_float('lambda', 0.0, 2.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        }
        model = XGBRegressor(random_state=42, **params)

    elif model_type == 'SVR':
        # Define the search for hyperparameters
        params = {
            'C': trial.suggest_loguniform('C', 1e-3, 1e3),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'epsilon': trial.suggest_uniform('epsilon', 0.01, 0.1)
        }
        model = SVR(**params)

    elif model_type == 'ElasticNet':
        # Define the search for hyperparameters
        params = {
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
            'l1_ratio': trial.suggest_uniform('l1_ratio', 0.0, 1.0)
        }
        model = ElasticNet(random_state=42, **params)

    elif model_type == 'DecisionTreeRegressor':
        # Define the search for hyperparameters
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 50),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
        }
        model = DecisionTreeRegressor(random_state=42, **params)
    else:
        raise ValueError("Invalid model type")

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_val)

    # Calculate mean squared error
    return mean_squared_error(y_val, y_pred)

def apply_PCA(train, val, test):
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the data.

    Parameters:
    - train (pd.DataFrame): Training data.
    - val (pd.DataFrame): Validation data.
    - test (pd.DataFrame): Test data.

    Returns:
    - tuple: Three DataFrames representing the transformed training, validation, and test sets after PCA.

    Example:
    >>> train_pca, val_pca, test_pca = apply_PCA(train_data, val_data, test_data)
    """

    # Initialize PCA with the desired explained variance
    pca = PCA(n_components=0.9)

    # Fit PCA on the training data
    train_pca = pca.fit_transform(train)

    # Transform the val/test data using the same PCA transformation
    val_pca = pca.transform(val)
    test_pca = pca.transform(test)

    # Convert the PCA-transformed data back to dataframes with the same structure
    train = pd.DataFrame(data=train_pca, index=train.index, columns=[f'PC_{i}' for i in range(1, train_pca.shape[1] + 1)])
    val = pd.DataFrame(data=val_pca, index=val.index, columns=[f'PC_{i}' for i in range(1, val_pca.shape[1] + 1)])
    test = pd.DataFrame(data=test_pca, index=test.index, columns=[f'PC_{i}' for i in range(1, test_pca.shape[1] + 1)])

    return train, val, test

def plot_results(y_train, y_val, y_test, y_pred, X_train, X_test, mape, model):
    """
    Plots the actual vs. predicted values for the test set and the entire time series.

    Parameters:
    - y_train (pd.Series): Actual values for the training set.
    - y_val (pd.Series): Actual values for the validation set.
    - y_test (pd.Series): Actual values for the test set.
    - y_pred (np.array): Predicted values for the test set.
    - X_train (pd.DataFrame): Features for the training set.
    - X_test (pd.DataFrame): Features for the test set.
    - mape (float): Mean absolute percentage error of the predictions.
    - model: Trained machine learning model.

    Returns:
    - None: Displays the plot.

    Example:
    >>> plot_results(y_train, y_val, y_test, y_pred, X_train, X_test, 5.2, trained_model)
    """

    # Create a subplot with two plots (2 rows, 1 column)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 14))

    # Plot the last month
    x = range(y_test.size)
    axes[0].plot(x, y_test, label='Actual', linewidth=2.0, color='steelblue')
    axes[0].plot(x, y_pred, label='Prediction', linewidth=2.0, color='orange')

    # Plot beautiful x-axis
    date_objects = [datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S") for date in X_test.index]
    formatted_dates = [date.strftime("%Y-%m-%d") for date in date_objects]
    axes[0].set_xticks(range(0, len(formatted_dates), 5))
    axes[0].set_xticklabels(formatted_dates[::5], rotation=45, ha='right')
    axes[0].set_title('Mean absolute percentage error {0:.2f}%'.format(mape))
    axes[0].legend(loc='best')
    axes[0].grid(True)

    # Plot the whole time series
    axes[1].plot(range(y_train.size), y_train, label='Historic', linewidth=2.0, color=(0.36, 0.73, 0.36))
    axes[1].plot(range(len(X_train) + y_val.size, len(X_train) + y_val.size + y_test.size), y_test,
                 label='Test Set', linewidth=2.0, color='steelblue')
    axes[1].plot(range(len(X_train) + y_val.size, len(X_train) + y_val.size + y_pred.size), y_pred,
                 label='Prediction', linewidth=2.0, color='orange')
    axes[1].plot(range(len(X_train), len(X_train) + y_val.size), y_val, label='Validation Set', linewidth=2.0,
                 color='gray')
    axes[1].plot(range(y_train.size), model.predict(X_train), label='Teaching Run', linewidth=2.0, color='red')

    # Plot beautiful x-axis
    date_objects = [datetime.strptime(str(date), "%Y-%m-%d %H:%M:%S") for date in X_train.index]
    formatted_dates = [date.strftime("%Y-%m-%d") for date in date_objects]
    axes[1].set_xticks(range(0, len(formatted_dates), 90))
    axes[1].set_xticklabels(formatted_dates[::90], rotation=45, ha='right')
    axes[1].set_title('Mean absolute percentage error {0:.2f}%'.format(mape))
    axes[1].legend(loc='best')
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

def plot_features_importance(model, X_train, name):
    """
    Plots the feature importance for a given machine learning model.

    Parameters:
    - model: Trained machine learning model with feature_importances_ attribute.
    - X_train (pd.DataFrame): Features used for training.
    - name (str): Name or label for the plot.

    Returns:
    - None: Displays the plot.

    Example:
    >>> plot_feature_importance(trained_model, X_train_data, 'Random Forest')
    """

    feature_importance = model.feature_importances_
    feature_names = X_train.columns

    # Print feature importance
    print("\nFeature Importance for", name)

    # Get the indices that would sort the feature_importance array in descending order
    sorted_idx = np.argsort(feature_importance)[::-1]

    # Visualize feature importance using Seaborn with ordered bars
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importance[sorted_idx], y=np.array(feature_names)[sorted_idx], palette="viridis")

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance for {name}")
    plt.show()

def evaluate_models(train, validation , test, plot_figures = True, plot_feature_importance = True, use_PCA = False, verbose_optuna = False):
    """
    Evaluates different machine learning models on a time series dataset.

    Parameters:
    - train (pd.DataFrame): Training data.
    - validation (pd.DataFrame): Validation data.
    - test (pd.DataFrame): Test data.
    - plot_figures (bool): Whether to plot the results.
    - plot_feature_importance (bool): Whether to plot feature importance.
    - use_PCA (bool): Whether to apply Principal Component Analysis (PCA) for dimensionality reduction.
    - verbose_optuna (bool): Whether to print Optuna optimization details.

    Returns:
    - tuple: Lists containing Mean Absolute Percentage Error (MAPE) and Root Mean Squared Error (RMSE) for each model.

    Example:
    >>> mape_list, rmse_list = evaluate_models(train_data, val_data, test_data, True, True, False, True)
    """
    
    # Create target variables
    X_train = train.drop(['count'], axis = 1)
    X_val = validation.drop(['count'], axis =1)
    X_test = test.drop(['count'], axis = 1)

    y_train = train['count']
    y_val = validation['count']
    y_test = test['count']

    # Standarize data
    scaler = StandardScaler()

    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the val/test data using the same scaler
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert the scaled arrays back to DataFrames with the original index
    X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_val = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
    X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    if use_PCA:
        # In this case, feature importance makes no sense
        plot_feature_importance = False

        # Apply PCA
        X_train, X_val, X_test = apply_PCA(X_train, X_val, X_test)
        

    # Define the models and tune the parameters
    models_name = ['RandomForest','GradientBoosting','XGBoost','SVR','ElasticNet','DecisionTreeRegressor','Voting Ensemble','Stacking Ensemble']
    mape_list = []
    rmse_list = []
    for name in models_name:
        print('\n\n==================',name,'===================\n\n')
        # Run Optuna optimization
        if name != 'Voting Ensemble' and name != 'Stacking Ensemble':
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Seed for reproducibility
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda trial: objective(trial, name, X_train,X_val,y_train,y_val), n_trials=100)

            if verbose_optuna:
                # Plot the optimization history
                optuna.visualization.plot_optimization_history(study).show()

                # Plot the parallel coordinate plot
                optuna.visualization.plot_parallel_coordinate(study).show()

            # Get the best parameters of the model
            best_trial = study.best_trial.params
            print('Best parameters:', best_trial)

        # Re-fit the model with the best parameters
        if name == 'RandomForest':
            model = RandomForestRegressor(random_state=42, **best_trial)
            params_rf = best_trial
        elif name == 'GradientBoosting':
            model = GradientBoostingRegressor(random_state=42, **best_trial)
        elif name == 'XGBoost':
            model = XGBRegressor(random_state=42, **best_trial)
            params_xgb = best_trial
        elif name == 'SVR':
            model = SVR(**best_trial)
            params_svr = best_trial
        elif name == 'ElasticNet':
            model = ElasticNet(random_state=42, **best_trial)
        elif name == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=42, **best_trial)
        elif name == 'Voting Ensemble':
            model = VotingRegressor(estimators=[('rf', RandomForestRegressor(random_state=42, **params_rf)),
                                                ('xgb', XGBRegressor(random_state=42, **params_xgb)),
                                                ('svr', SVR( **params_svr))])
        elif name == 'Stacking Ensemble':
            model = StackingRegressor(estimators=[('rf', RandomForestRegressor(random_state=42, **params_rf)),
                                                ('svr', SVR(**params_svr))], final_estimator=XGBRegressor(random_state=42,**params_xgb))
            
        else:
            raise ValueError("Invalid model type")
       
        # Train the model
        model.fit(X_train, y_train)

        # Get predictions on the test
        y_pred = model.predict(X_test)# This is equivalent to make a loop and do it separately for each day

        # Compute metrics
        mape = mean_absolute_percentage_error(y_test, y_pred)*100
        rmse = mean_squared_error(y_test,y_pred)**.5
        print('MAPE=',mape)
        print('RMSE=',rmse)
        # Append the results
        mape_list.append(mape)
        rmse_list.append(rmse)
        

        # Plot results
        if plot_figures:
            plot_results(y_train, y_val, y_test, y_pred, X_train, X_test, mape, model)

            
            # Print or visualize feature importance
            if hasattr(model, 'feature_importances_') and plot_feature_importance:
                plot_features_importance(model, X_train, name)
        
    return mape_list, rmse_list

def parameters_search(df, max_lag, max_rolling_window):
    """
    Perform a grid search for lag and rolling window parameters, evaluating models for each combination.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing time series data.
    - max_lag (int): Maximum lag value for grid search.
    - max_rolling_window (int): Maximum rolling window value for grid search.

    Returns:
    - dict: Results of the parameter search, including minimum MAPE, RMSE, and average values, along with corresponding parameters.

    Example:
    >>> search_results = parameters_search(df_sensor_data, 5, 10)
    """

    # Peform train/val/test split
    train, valid, test = train_val_test_split_ts(df)

    # Generate grid search for lag and window parameters
    grid_search_parameters = []

    for lag in range(0, max_lag + 1):
        for window in range(2, max_rolling_window + 1):
            window_list = list(range(2, window + 1))
            grid_search_parameters.append({'lag': lag, 'window': window_list})

 

    # Initialize variables to keep track of the minimum values
    min_mape = float('inf')
    min_rmse = float('inf')
    min_avg_mape = float('inf')
    min_avg_rmse = float('inf')
    best_mape_params = None
    best_rmse_params = None
    best_avg_mape_params = None
    best_avg_rmse_params = None
    
    # Generate grid search of parameters and find the best combination
    print('Starting parameter gridsearch...')
    for params in grid_search_parameters:
        print('\n\nLags=',params['lag'])
        print('Window=',params['window'])


        # Create the lagged dataset
        train_lag = add_lags(train, params['lag'], 'count')
        valid_lag = add_lags(valid, params['lag'] ,'count')
        test_lag = add_lags(test, params['lag'], 'count')

        # Create the final dataset
        train_all = create_rolling_features(train_lag, 'count', windows= params['window'])
        valid_all = create_rolling_features(valid_lag, 'count', windows= params['window'])
        test_all = create_rolling_features(test_lag, 'count', windows = params['window'])

        # Train models
        mape, rmse = evaluate_models(train_all, valid_all, test_all, plot_figures=False, use_PCA= False)

        # Update minimum values for each model
        length = len(mape)
        for i in range(length):
            if mape[i] < min_mape:
                min_mape = mape[i]
                best_mape_params = params
            if rmse[i] < min_rmse:
                min_rmse = rmse[i]
                best_rmse_params = params

        # Calculate and update average values for each model
        avg_mape = np.mean(mape)
        avg_rmse = np.mean(rmse)

        if avg_mape < min_avg_mape:
            min_avg_mape = avg_mape
            best_avg_mape_params = params
        if avg_rmse < min_avg_rmse:
            min_avg_rmse = avg_rmse
            best_avg_rmse_params = params
            
    # Return the parameters        
    result = {
        'min_mape': min_mape,
        'best_mape_params': best_mape_params,
        'min_rmse': min_rmse,
        'best_rmse_params': best_rmse_params,
        'min_avg_mape': min_avg_mape,
        'best_avg_mape_params': best_avg_mape_params,
        'min_avg_rmse': min_avg_rmse,
        'best_avg_rmse_params': best_avg_rmse_params,
    }

    return result

def create_best_combination_dataset(df, lag, window):
    """
    Create a dataset with the best combination of lag and rolling window parameters.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing time series data.
    - lag (int): Lag value for creating lag features.
    - window (int): Rolling window size for creating rolling features.

    Returns:
    - tuple: Train, validation, and test datasets with lag and rolling window features.

    Example:
    >>> train_data, valid_data, test_data = create_best_combination_dataset(df_sensor_data, 3, [2,3,4])
    """

    # Train/valid/test split
    train, valid, test = train_val_test_split_ts(df)

    # Create lagged dataset
    train_lag = add_lags(train, lag, 'count')
    valid_lag = add_lags(valid, lag, 'count')
    test_lag = add_lags(test, lag, 'count')

    # Create final dataset
    train_all = create_rolling_features(train_lag, 'count', windows=window)
    valid_all = create_rolling_features(valid_lag, 'count', windows=window)
    test_all = create_rolling_features(test_lag, 'count', windows=window)

    return train_all, valid_all, test_all

def all_sensors_data(df):
    """
    Process sensor data to create a DataFrame with date as index, IP addresses as columns, and counts as values.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing sensor data.

    Returns:
    - pd.DataFrame: Processed DataFrame with date as index, IP addresses as columns, and counts as values.

    Example:
    >>> processed_data = all_sensors_data(df_sensor_data)
    """

    # Convert 'startTime' to datetime
    df['date'] = pd.to_datetime(df['_source.startTime'])

    # Extract the date (day) from the timestamp
    df['date'] = pd.to_datetime(df['date'].dt.date)

    # Group by 'date' and 'IP', then count occurrences
    grouped_df = df.groupby(['date', '_source.hostIP']).size().reset_index(name='count')

    # Pivot the table to have 'date' as index, 'IP' as columns, and 'count' as values
    result_df = grouped_df.pivot_table(index='date', columns='_source.hostIP', values='count', fill_value=0)

    return result_df

def merge_all_sensors(df, df_country, lags):
    """
    Merge sensor data with country data and add lags for each sensor.

    Parameters:
    - df (pd.DataFrame): Sensor data DataFrame with date as index and sensors as columns.
    - df_country (pd.DataFrame): Country data DataFrame with date as index and countries as columns.
    - lags (int): Number of lags to add for each sensor.

    Returns:
    - pd.DataFrame: Merged DataFrame with lags for each sensor and country data.

    Example:
    >>> merged_data = merge_all_sensors(df_sensor_data, df_country_data, 3)
    """

    # Add lags of all sensors
    for i in range(1, lags + 1):
        df_lag = df.shift(i)
        if i == 1:
            df_result = pd.merge(df_lag, df_country, left_index=True, right_index=True)
        else:
            df_result = pd.merge(df_lag, df_result, left_index=True, right_index=True)

    df_result.fillna(0, inplace=True)
    return df_result




#---------------------- Deep learning methods ----------------------#