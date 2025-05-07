import pandas as pd
import matplotlib.pyplot as plt                                            
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from matplotlib.ticker import MaxNLocator
from scipy.signal import find_peaks
from adjustText import adjust_text

def keep_rows(df, conditions):
    for column, values in conditions.items():
        df = df[df[column].isin(values)]
    return df

def filter_dates_between(df, start_date, end_date, date_column='data_inversa'):
    """
    Filters the DataFrame to keep only rows with dates between start_date and end_date.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the date column or with a date index.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    date_column (str, optional): The name of the date column in the DataFrame. Default is 'data_inversa'.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    if date_column == 'index':
        # Ensure the index is a datetime object
        for date_format in ['%d %b %Y', '%d/%m/%Y','%d-%b-%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d']:
            try:
                date_col = pd.to_datetime(df.index, format=date_format)
                break

            except:
                continue
        
        #df.index = date_col

    else:
        # Convert the date column to datetime format
        for date_format in ['%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d']:
            try:
                date_col = pd.to_datetime(df[date_column], format=date_format)
                break
            except:
                continue
        
        #df[date_column] = date_col
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the DataFrame
    filtered_df = df[(date_col >= start_date) & (date_col <= end_date)]

    return filtered_df
    
    # Convert start_date and end_date to datetime format
    

def DecomposedTrendDataset(TimeSeries_Dataset,seasonal=13,period=12):
    # Calculate the score and moran's index
    trend_components = {}
    seasonal_components = {}
    resid_components = {}
    for column in TimeSeries_Dataset.columns:
        stl = STL(TimeSeries_Dataset[column], seasonal=seasonal, period=period)
        result = stl.fit()
        
        # Store the trend component in the dictionary
        trend_components[column] = result.trend
        seasonal_components[column] = result.seasonal
        resid_components[column] = result.resid
        
    trend_Dataset = pd.DataFrame(trend_components)
    seasonal_Dataset = pd.DataFrame(seasonal_components)
    resid_Dataset = pd.DataFrame(resid_components)
    return trend_Dataset, seasonal_Dataset, resid_Dataset

def CreateTimeSeriesDataset(df,attribute,values, granularity=3,count='months'):
    new_df = create_daily_timseries_dataset(df,attribute,values)
    
    if new_df is None:
        return None    
    else:
        new_df = sort_time_series(new_df)
    
        df_resampled = granular_resample(new_df,count,granularity)
    
        return df_resampled

def create_daily_timseries_dataset(df,attribute,values):
    time_series_list = []
    found = []
    for value in values:
        new_df = keep_rows(df, {attribute: [value]})
        value_timeSeries = create_time_series(new_df)
        if not value_timeSeries.empty:
            time_series_list.append(value_timeSeries)
            found.append(value)
        
    # Create the dataset from all formed time-series
    if time_series_list:
        new_df = pd.concat(time_series_list, axis=1,join='outer')
        new_df.columns = found
        new_df = new_df.fillna(0)
        return new_df
    # If no location had occurences of this class
    else: return None
    
    

def granular_resample(ts,count,granularity):
    
    # Determine the resampling frequency
    if count == 'days':
        frequency = f'{granularity}D'
    elif count == 'months':
        frequency = f'{granularity}ME'
    else:
        raise ValueError("Count must be either 'days' or 'months'")
        
    # Resample the DataFrame to the desired frequency and sum the values
    ts_resampled = ts.resample(frequency).sum()
    ts_resampled.index = ts_resampled.index.strftime('%d %b %Y')
    
    return ts_resampled

def sort_time_series(ts):
    ts = ts.copy()
    # SORT TIME SERIES
    ts.index = pd.to_datetime(ts.index,format='%d %b %Y')
    ts.index = ts.index.strftime('%y %m %d')
    ts.sort_index(inplace=True)
    ts.index = pd.to_datetime(ts.index,format='%y %m %d')
    
    return ts

def create_time_series(df):
    df = df.copy() 
    # Drop rows where 'data_inversa' is not a valid date
    df['data_inversa'] = pd.to_datetime(df['data_inversa'], format='%d/%m/%Y', errors='coerce')
    df.dropna(subset=['data_inversa'],inplace=True)
    
    # Convert 'data_inversa' to the specified format '%d %b %Y'
    df['data_inversa'] = df['data_inversa'].dt.strftime('%d %b %Y')
    # Create a time series with formatted dates and their counts
    time_series = df['data_inversa'].value_counts()
    return time_series


def splitTimeSeries(timeSeries, reference_dates):
    # Define lockdown dates
    date1 = pd.to_datetime(reference_dates[0])
    date2 = pd.to_datetime(reference_dates[1])
    
    # Ensure the index is datetime for slicing
    timeSeries.index = pd.to_datetime(timeSeries.index)
    
    # Split the DataFrame into three ranges
    range1 = timeSeries[timeSeries.index < date1]
    range2 = timeSeries[(timeSeries.index >= date1) & (timeSeries.index <= date2)]
    range3 = timeSeries[timeSeries.index > date2]

    # Concatenate the ranges and set column names
    new_df = pd.concat([range1, range2, range3], axis=1, join='outer')
    new_df.columns = ['Before lockdown', 'During lockdown', 'After lockdown']
    
    # Convert index back to desired format for display
    new_df.index = new_df.index.strftime('%d %b %Y')
    
    return new_df
    

    
def plot_timeSeriesDecomposed( original_Dataset, trend_Dataset, seasonal_Dataset, resid_Dataset, reference_dates = None):

    # For each time series of the dataframe:
    for column in original_Dataset:
        
        original_TS = original_Dataset[column]
        trend_TS = trend_Dataset[column]
        seasonal_TS = seasonal_Dataset[column]
        resid_TS = resid_Dataset[column]
        
        ymax = max(original_TS)*1.2
        ymin = min(original_TS)*0.8
        
        # Set up the figure and subplots
        plt.figure(figsize=(12, 18))
        
        plt.subplot(4, 1, 1)
        sns.lineplot(data=splitTimeSeries(original_TS,reference_dates), palette='rocket')
        for line in plt.gca().get_lines(): line.set_linestyle('-')  # Set each line's style to solid
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
        #plt.title(f'Time Series of {attribute}')
        plt.gca().set_facecolor('#f0f0f0')
        plt.ylim(ymin, ymax)
        
        plt.subplot(4, 1, 2)
        sns.lineplot(data=splitTimeSeries(trend_TS,reference_dates), palette='rocket')
        for line in plt.gca().get_lines(): line.set_linestyle('-')  # Set each line's style to solid
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
        #plt.title(f'Time Series of {attribute}')
        plt.gca().set_facecolor('#f0f0f0')
        plt.ylim(ymin, ymax)
        
        # Peaks and Valleys in the tred:
        
        peaks, _ = find_peaks(trend_TS)
        valleys, _ = find_peaks(-trend_TS)
        peak_dates = trend_TS.iloc[peaks].index.strftime('%d %b %Y')
        peak_values = trend_TS.iloc[peaks]
        valley_dates = trend_TS.iloc[valleys].index.strftime('%d %b %Y')
        valley_values = trend_TS.iloc[valleys]


        
        # Plot the peaks
        plt.scatter(peak_dates, peak_values, color='royalblue', label='Peaks', s=100)
        plt.scatter(valley_dates, valley_values, color='limegreen', label='Peaks', s=100)
        
        """
        texts = [plt.text(date, value, date, ha='center', va='bottom', fontsize=8, color='black') 
                 for date, value in zip(peak_dates, peak_values)]
        
        adjust_text(texts, expand=(3, 3), arrowprops=dict(arrowstyle='->', color='royalblue'))
        
        texts = [plt.text(date, value, date, ha='center', va='bottom', fontsize=8, color='black') 
                 for date, value in zip(valley_dates, valley_values)]
        
        adjust_text(texts, expand=(3, 3), arrowprops=dict(arrowstyle='->', color='limegreen'))
        """
            
        
        plt.subplot(4, 1, 3)
        sns.lineplot(data=splitTimeSeries(seasonal_TS,reference_dates), palette='rocket')
        for line in plt.gca().get_lines(): line.set_linestyle('-')  # Set each line's style to solid
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
        #plt.title(f'Time Series of {attribute}')
        plt.gca().set_facecolor('#f0f0f0')
        plt.ylim(-(ymax-ymin)/2, (ymax-ymin)/2)
        
        plt.subplot(4, 1, 4)
        sns.lineplot(data=splitTimeSeries(resid_TS,reference_dates), palette='rocket')
        for line in plt.gca().get_lines(): line.set_linestyle('-')  # Set each line's style to solid
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=7))
        #plt.title(f'Time Series of {attribute}')
        plt.gca().set_facecolor('#f0f0f0')
        plt.ylim(-(ymax-ymin)/2, (ymax-ymin)/2)
        # Calculate mean and standard deviation for residuals

        
        # Show plot
        plt.tight_layout()
        plt.rcParams["figure.figsize"] = (12, 18)
        plt.gcf().set_dpi(300)
        plt.savefig("Trend_timeSeries_byCategory_" + column.replace('/', ' ') + '.png', bbox_inches='tight')
        plt.show()
        
        
        
    
