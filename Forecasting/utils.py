# General functions for the Forecasting project
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
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
    plt.legend(loc='best')
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








#---------------------- Machine learning methods ----------------------#










#---------------------- Deep learning methods ----------------------#