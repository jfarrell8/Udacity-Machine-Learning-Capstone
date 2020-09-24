import pandas as pd
import matplotlib.pyplot as plt
import json

def make_time_series(df, years, frequency_change=False, freq=None):
    """
    Creates time series of adjusted closing stock prices for every year provided.
    :param df: A dataframe that contains daily historical stock prices for a given stock.
    :param years: A list of years that will be used to make a time series, ex. ['2016',...,'2018']
    :param frequency_change: set to True if looking to switch from daily prices to different frequency
            of price.
    :param freq: set to None by default. Set to M' for a monthly resampling or 'Y' for a yearly resampling.
    """
    
    # store time series
    time_series = []
    
    # if a frequency change of the data is desired...resample
    if frequency_change:
        df = df.resample(freq).mean()
    
    # create time series for each year in years
    for i in range(len(years)):

        year = years[i]
        time_series.append(pd.Series(df.loc[year]['Adj. Close'], index=df.loc[year].index))
        
    return time_series

def range_to_years(start_year, end_year):
    """
    Convert start and end dates provided to a list of years. List must be strings.
    """
    
    years = []
    yr_timestamp = pd.date_range(start_year, str(int(end_year)+1), freq='Y').to_list()
    years = [str(i)[:4] for i in yr_timestamp] 
        
    return years

# create truncated, training time series
def create_training_series(complete_time_series, prediction_length):
    """
    Given a complete list of time series data, create training time series.
       :param complete_time_series: A list of all complete time series.
       :param prediction_length: The number of points we want to predict.
       :return: A list of training time series.
    """
    
    # your code here
    time_series_training = []
    
    for ts in complete_time_series:
        time_series_training.append(ts[:-prediction_length])
        
    return time_series_training

def series_to_json_obj(ts):
    """
    Returns a dictionary of values in DeepAR, JSON format.
       :param ts: A single time series.
       :return: A dictionary of values with "start" and "target" keys.
    """
    
    # get start time and target from the time series, ts
    json_obj = {"start": str(ts.index[0]), "target": list(ts)}
    return json_obj

def write_json_dataset(time_series, filename): 
    """
    Writes time series to a filename(path).
      :param time_series: Single time series.
      :filename: path to save locally
    """
    with open(filename, 'wb') as f:
    # for each of our times series, there is one JSON line
        for ts in time_series:
            json_line = json.dumps(series_to_json_obj(ts)) + '\n'
            json_line = json_line.encode('utf-8')
            f.write(json_line)
    print(filename + ' saved.')
    
def json_predictor_input(input_ts, num_samples=50, quantiles=['0.1', '0.5', '0.9']):
    """
    Accepts a list of input time series and produces a formatted input.
       :input_ts: An list of input time series.
       :num_samples: Number of samples to calculate metrics with.
       :quantiles: A list of quantiles to return in the predicted output.
       :return: The JSON-formatted input.
    """
    # request data is made of JSON objects (instances)
    # and an output configuration that details the type of data/quantiles we want
    
    instances = []
    for k in range(len(input_ts)):
        # get JSON objects for input time series
        instances.append(series_to_json_obj(input_ts[k]))

    # specify the output quantiles and samples
    configuration = {"num_samples": num_samples, 
                     "output_types": ["quantiles"], 
                     "quantiles": quantiles}

    request_data = {"instances": instances, 
                    "configuration": configuration}

    json_request = json.dumps(request_data).encode('utf-8')
    
    return json_request

# helper function to decode JSON prediction
def decode_prediction(prediction, encoding='utf-8'):
    """
    Accepts a JSON prediction and returns a list of prediction data.
    """
    
    prediction_data = json.loads(prediction.decode(encoding))
    prediction_list = []
    for k in range(len(prediction_data['predictions'])):
        prediction_list.append(pd.DataFrame(data=prediction_data['predictions'][k]['quantiles']))
    return prediction_list

# display the prediction median against the actual data
def display_quantiles(prediction_list, prediction_length, target_ts=None):
    """
    Display average prediction value with 80% confidence interval compared to
    target values.
      :param prediction_list: list of predictions for stock prices over time at 0.1, 0.5, 0.9 quantiles
      :prediction_length: how far in the future we are trying to predict
      :target_ts: the target time series to compare against
    """
    
    # show predictions for all input ts
    for k in range(len(prediction_list)):
        plt.figure(figsize=(12,6))
        # get the target month of data
        if target_ts is not None:
            target = target_ts[k][-prediction_length:]
            plt.plot(range(len(target)), target, label='target')
        # get the quantile values at 10 and 90%
        p10 = prediction_list[k]['0.1']
        p90 = prediction_list[k]['0.9']
        # fill the 80% confidence interval
        plt.fill_between(p10.index, p10, p90, color='y', alpha=0.5, label='80% confidence interval')
        # plot the median prediction line
        prediction_list[k]['0.5'].plot(label='prediction median')
        plt.legend()
        plt.show()