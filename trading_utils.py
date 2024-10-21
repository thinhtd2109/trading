import numpy as np
from datetime import datetime, timedelta
import dateutil.parser
import pandas as pd
import ta
import ta.momentum
import ta.trend
from ta import add_all_ta_features
from joblib import load
from typing import Dict, Any

def time_difference(t1, t2):
    """Calculate the minimal difference between two times, considering circularity."""
    delta1 = abs(datetime.combine(datetime.min, t1) - datetime.combine(datetime.min, t2))
    delta2 = timedelta(hours=23, minutes=59, seconds=59) - delta1
    return min(delta1, delta2)

def get_weekday(date_str): 
    date = dateutil.parser.isoparse(date_str)
    weekday_number = date.weekday() 
    return weekday_number + 2


def preprocess_input(data, scalers):
    """ Preprocess the input JSON data to a format suitable for the model """
    time_step = 20
    features = ["STOCH"]
    num_features = len(features)
    input_data = np.zeros((1, time_step, num_features))
    
    # Apply individual scaling and format to input shape
    for i, feature in enumerate(features):
        values = data[feature]
        scaled_values = scalers[feature].transform(np.array(values).reshape(-1, 1)).flatten()
        input_data[0, :, i] = scaled_values
    
    return input_data 

def predict_indicator(new_data, rf_classifier, quantity, indicator):
    features = []
    new_data['Price_Change'] = round(new_data['close'].pct_change(), 5)  # % thay đổi giá
    new_data['Candle_Length'] = round(new_data['close'] - new_data['open'] , 5) # Chiều dài của nến
    new_data['Volatility'] = round(new_data['high'] - new_data['low'], 5)  # Biến động giá
    if indicator != 'NORMAL':
        # Create lagged features
        shifted_columns = [new_data[f'{indicator}'].shift(i) for i in range(0, quantity)]
        shifted_df = pd.concat(shifted_columns, axis=1)
        shifted_df.columns = [f'{indicator}_{i}' for i in range(0, quantity)]
        
        # Combine original DataFrame with shifted columns
        new_data = pd.concat([new_data, shifted_df], axis=1)

        # Prepare features for prediction
        features = [f'{indicator}_{i}' for i in range(0, quantity)] 
    else:
        features = ['Price_Change', 'Candle_Length', 'Volatility']
    # Get the latest data for prediction
    latest_data = new_data[features].iloc[-1].to_frame().T

    # Predict the next price movement
    next_movement = rf_classifier.predict(latest_data)


    return int(next_movement)


# # Dictionary mapping indicator names to model filenames
# model_filenames = {
#     'STOCH': './random_forest_xauusd_model_STOCH.pkl',
#     'TSI': './random_forest_xauusd_model_TSI.pkl',
#     'RSI': './random_forest_xauusd_model_RSI.pkl',
#     'WILL': './random_forest_xauusd_model_WILL.pkl',
#     'CCI': './random_forest_xauusd_model_CCI.pkl',
#     'OSC': './random_forest_xauusd_model_OSC.pkl',
#     'STOCH_RSI': './random_forest_xauusd_model_STOCH_RSI.pkl',
#     'DPO': './random_forest_xauusd_model_DPO.pkl',
#     'NORMAL': './random_forest_xauusd_model_NORMAL.pkl'
# }

# # Load all models using a loop
# rf_classifiers = {key: load(path) for key, path in model_filenames.items()}


# def calculate_indicators(new_data, round_num=5):
#     """Calculate technical indicators and add them to the DataFrame."""
#     new_data['body'] = np.round(new_data['close'] - new_data['open'], round_num)
    
#     indicators = {
#         'STOCH': ta.momentum.StochasticOscillator(
#             high=new_data['high'], low=new_data['low'], close=new_data['close']).stoch(),
#         'TSI': ta.momentum.TSIIndicator(close=new_data['close']).tsi(),
#         'RSI': ta.momentum.RSIIndicator(close=new_data['close']).rsi(),
#         'WILL': ta.momentum.WilliamsRIndicator(
#             close=new_data['close'], high=new_data['high'], low=new_data['low']).williams_r(),
#         'CCI': ta.trend.CCIIndicator(
#             close=new_data['close'], high=new_data['high'], low=new_data['low']).cci(),
#         'OSC': ta.momentum.AwesomeOscillatorIndicator(
#             high=new_data['high'], low=new_data['low']).awesome_oscillator(),
#         'STOCH_RSI': ta.momentum.StochRSIIndicator(close=new_data['close']).stochrsi(),
#         'DPO': ta.trend.DPOIndicator(close=new_data['close']).dpo()
#     }

#     # Round the indicators
#     for key, value in indicators.items():
#         new_data[key] = np.round(value, round_num)

#     return new_data


# def predict_all_indicators(new_data, quantity=20):
#     """Predict the direction for each indicator."""
#     predictions = {}
#     for indicator, model in rf_classifiers.items():
#         predictions[indicator] = predict_indicator(
#             new_data=new_data, rf_classifier=model, quantity=quantity, indicator=indicator
#         )
#     return predictions


# def predict_next_price():
#     round_num = 5
#     quantity = 20
#     num_rates = 200
#     data = request.get_json()
#     start = int(data.get('start', 1)) 
    
#     # Fetch historical rates
#     rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, num_rates)
#     new_data = pd.DataFrame(rates)


    
#     # Calculate indicators
#     new_data = calculate_indicators(new_data=new_data, round_num=round_num)

#     # Predict based on all indicators
#     predictions = predict_all_indicators(new_data=new_data, quantity=quantity)
    
#     accumulate = list(predictions.values())
#     print(accumulate)
    
#     sells = accumulate.count(0)
#     buys = accumulate.count(1)

#     next_predict = 1 if buys > sells else 0
#     if buys == sells:
#         next_predict = None

#     # Return the predicted price
#     return jsonify({'predicted_next_closing_price': next_predict})

def create_segments(data, window_size=10, indicators = []):
    return np.array([
        np.hstack([data[col].values[i:i + window_size] for col in indicators])
        for i in range(len(data) - window_size + 1)
    ])


def predict_nearest(window_size, new_data, df_historical, model, indicator):

    historical_segments = create_segments(data=new_data, window_size=window_size, indicators=indicator)[-1]

    distances, indices = model.kneighbors([historical_segments])

    print(indices)

    most_index = indices[0][0]
    current_candle = df_historical.iloc[most_index]
    return int(current_candle['target'])

def processing_data(new_data):
    round_num = 5
    range_value = 10
    new_data['Volume'] = new_data['tick_volume']
    STOCH = ta.momentum.StochasticOscillator(new_data['high'], new_data['low'], new_data['close'], fillna=True)
    TSI = ta.momentum.TSIIndicator(close=new_data['close'])
    RSI = ta.momentum.RSIIndicator(close=new_data['close'])
    STOCH_RSI = ta.momentum.StochRSIIndicator(close=new_data['close'])
    new_data['STOCH_RSI'] = np.round(STOCH_RSI.stochrsi(), round_num)
    new_data['TSI'] = np.round(TSI.tsi(), round_num)
    new_data['RSI'] = np.round( RSI.rsi(), round_num)
    new_data['STOCH'] = np.round(STOCH.stoch(), round_num)
    lags = range(0, range_value)

    columns = [new_data['STOCH'].shift(i).rename(f'STOCH_{i}') for i in lags]
    df = pd.concat(columns, axis=1)
    new_data = new_data.join(df)

    # Kết hợp DataFrame mới với DataFrame gốc
    columns = [new_data['TSI'].shift(i).rename(f'TSI_{i}') for i in lags]
    df = pd.concat(columns, axis=1)
    new_data = new_data.join(df)

    # Kết hợp DataFrame mới với DataFrame gốc
    columns = [new_data['RSI'].shift(i).rename(f'RSI_{i}') for i in lags]
    df = pd.concat(columns, axis=1)
    new_data = new_data.join(df)

    # Kết hợp DataFrame mới với DataFrame gốc
    columns = [new_data['STOCH_RSI'].shift(i).rename(f'STOCH_RSI_{i}') for i in lags]
    df = pd.concat(columns, axis=1)
    new_data = new_data.join(df)
    # new_data = add_all_ta_features(df=new_data, open="open", high="high", low="low", close="close", volume="Volume", fillna=True)    
    return new_data.dropna()
