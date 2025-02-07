import MetaTrader5 as mt5
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import ta 
import ta.momentum
import ta.trend
import ta.volatility
from joblib import load
from datetime import timedelta
import random

import ta.volume
from tensorflow.keras.models import load_model

# Load the model and scalers
model = load_model('./predict_model_XAUUSD.h5')
app = Flask(__name__) 


login = 182893487  
password = "Ducthinh@2109"
server = 'Exness-MT5Trial6'

# Khởi tạo MetaTrader 5 
if not mt5.initialize():
    print("Initialization failed",  mt5.last_error())
    mt5.shutdown()
    exit()

# login = 103865038
# password = "Ducthinh@2109" 
# server = "Exness-MT5Real15" 

symbol = "XAUUSD"   
isReverse = False  

def calculate_shadow_top(row):
    if row['close'] > row['open']:  # Nến tăng
        return row['high'] - row['close']
    else:  # Nến giảm
        return row['high'] - row['open']

def calculate_shadow_bottom(row):
    if row['close'] > row['open']:  # Nến tăng
        return row['open'] - row['low']
    else:  # Nến giảm
        return row['close'] - row['low']


if not mt5.login(login=login, password=password, server=server):  
    print("Login failed")
    mt5.shutdown() 
    exit() 

# def create_segments(data, window_size, columns):
#     return np.array([
#         np.hstack([data[col].values[i:i + window_size] for col in columns])
#         for i in range(len(data) - window_size + 1)
#     ])

def create_segments(data: pd.DataFrame, columns: list, window_size: int):
    segments = []
    num_rows = len(data)
    for i in range(num_rows - window_size + 1):
        segment_values = []
        for col in columns:
            # Lấy 15 giá trị liên tiếp của cột col
            segment_values.extend(data[col].values[i:i + window_size])
        segments.append(segment_values)
    return np.array(segments)
# df_historical = pd.read_csv(f'./train-data/OANDA_XAUUSD_Historical.csv')
# # List of indicators
# indicators = ['AO', 'TSI', 'RSI', 'PPO', 'STOCH_RSI', 'STOCH', 'UO', 'WR', 'ROC']

# # Dictionary to store the models
# models = {}

# rf_classifier = load('./model_XAUUSD.pkl')

# # Loop to load models dynamically
# for indicator in indicators:
#     model_name = f'nearest_neighbors_model_XAUUSD_{indicator}.joblib'
#     models[indicator] = load(model_name)

@app.route('/predict', methods=['POST'])
def predict_next_price(): 

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, 100)
    new_data = pd.DataFrame(rates)

    # Calculate STOCH and other features
    STOCH = ta.momentum.StochasticOscillator(close=new_data['close'], high=new_data['high'], low=new_data['low'], fillna=True)
    new_data['stoch'] = STOCH.stoch()
    
    new_data['shadow_top'] = new_data.apply(calculate_shadow_top, axis=1)
    new_data['shadow_bottom'] = new_data.apply(calculate_shadow_bottom, axis=1)
    new_data['body'] = new_data['close'] - new_data['open']
    new_data['volume_avg'] = new_data['tick_volume'].rolling(window=15).mean()
    # Create lagged features
    lags = range(0, 15)
    features = []
    for col in ['body', 'shadow_top',  'shadow_bottom']:
        for lag in lags:
            new_data[f'{col}_{lag}'] = new_data[col].shift(lag)
            features.append(f'{col}_{lag}')
    
    # # Drop NaNs
    new_data.dropna(inplace=True)
    
    # # Prepare latest data for prediction
    next_closing_price = model.predict(np.array(new_data[features].iloc[-1]).reshape(1, 1, len(features)))

    next_moving_price = 1 if next_closing_price[0, 0] > 0.5 else 0

    return jsonify({'predicted_next_closing_price': next_moving_price })
    
# def predict_next_price(): 

#     rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, 15)
#     new_data = pd.DataFrame(rates)

#     # Calculate STOCH and other features
#     STOCH = ta.momentum.StochasticOscillator(close=new_data['close'], high=new_data['high'], low=new_data['low'], fillna=True)
#     new_data['stoch'] = STOCH.stoch()
    
#     new_data['shadow_top'] = new_data.apply(calculate_shadow_top, axis=1)
#     new_data['shadow_bottom'] = new_data.apply(calculate_shadow_bottom, axis=1)
#     new_data['body'] = new_data['close'] - new_data['open']
#     new_data['volume_avg'] = new_data['tick_volume'].rolling(window=15).mean()
#     # Create lagged features
#     lags = range(0, 15)
#     features = []
#     for col in ['body', 
#                 'shadow_top', 
#                 'shadow_bottom'
#         ]:
#         for lag in lags:
#             new_data[f'{col}_{lag}'] = new_data[col].shift(lag)
#             features.append(f'{col}_{lag}')
    
#     # # Drop NaNs
#     new_data.dropna(inplace=True)
    
#     # # Prepare latest data for prediction
#     latest_data = pd.DataFrame([new_data[features].iloc[-1]], columns=features)
#     next_movement = rf_classifier.predict(latest_data)

#     return jsonify({'predicted_next_closing_price': int(next_movement[0])})
    

@app.route('/get-open', methods=['POST'])   
def predict_next_price_v2():
    data = request.get_json()
    start = int(data.get('start', 1))
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M10, start, 5) 
    new_data = pd.DataFrame(rates)
    new_data['body'] = new_data['close'] - new_data['open']
    latest_candle = new_data.iloc[-1]
    return jsonify({'open': latest_candle['open'], 'body': latest_candle['body'] }) 

@app.route('/export', methods=['GET']) 
def export_csv():
    data = request.get_json()
    symbol = data['symbol']
    start = int(data['start'])
    end = int(data['end'])
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, start, end)
    
    # Convert the rates to a DataFrame
    rates_frame = pd.DataFrame(rates)
     
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s') + timedelta(hours=7)
    rates_frame['time'] = rates_frame['time'].dt.strftime("%Y-%m-%dT%H:%M:%S+07:00")
    rates_frame['volume'] = rates_frame['tick_volume']
    rates_frame['close'] = np.round(rates_frame['close'], 5)
    rates_frame['open'] = np.round(rates_frame['open'], 5)
    rates_frame['low'] = np.round(rates_frame['low'], 5)
    rates_frame['high'] = np.round(rates_frame['high'], 5)
    rates_frame = rates_frame[['time', 'open', 'high', 'low', 'close', 'volume']]

 
    # Export the DataFrame to a CSV file
    rates_frame.to_csv(f'./data_{symbol}.csv', index=False)
    
    # Return a JSON response indicating success
    return jsonify({'message': 'Export Successfully'})


@app.route('/klines', methods=['GET'])
def get_klines():
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, 100)
    rates_frame = pd.DataFrame(rates)
    rates_frame['shadow_top'] = rates_frame.apply(calculate_shadow_top, axis=1)
    rates_frame['shadow_bottom'] = rates_frame.apply(calculate_shadow_bottom, axis=1)
    rates_frame['body'] = rates_frame['close'] - rates_frame['open']
    rates_frame['body'] = rates_frame['close'] - rates_frame['open']
    rates_json = rates_frame.to_dict(orient='records') 
    # Return the prediction in JSON format
    return jsonify(rates_json) 

@app.route('/positions', methods=['GET'])
def get_positions():
    positions = mt5.positions_get(symbol='EURUSDm')     
    if not positions:
        return jsonify({ 'data': None })
    df=pd.DataFrame(list(positions),columns=positions[0]._asdict().keys())
    json = df.to_dict(orient='records')
    return jsonify(json[-1])

 
@app.route('/close-current-position', methods=['POST'])
def close_current_position():
    data = request.get_json()
    lot = data['lot']
    positions = mt5.positions_get(symbol=symbol)     
    if positions:
        last_position = positions[-1]
        if last_position.type == mt5.ORDER_TYPE_BUY:
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "position": last_position.ticket,
                "price": mt5.symbol_info_tick(symbol).bid,
                #"deviation": 20,
                "magic": 202003,
                "comment": "",
                "type_time": mt5.ORDER_TIME_GTC,
                #"type_filling": mt5.ORDER_FILLING_IOC
            }
            mt5.order_send(close_request)
        if last_position.type == mt5.ORDER_TYPE_SELL:
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol, 
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY,
                "position": last_position.ticket,
                "price": mt5.symbol_info_tick(symbol).ask,
            # "deviation": 20,
                "magic": 202003,
                "comment": "",
                "type_time": mt5.ORDER_TIME_GTC,
                #"type_filling": mt5.ORDER_FILLING_IOC
            }
            mt5.order_send(close_request)
    return jsonify({'message': 'Close Position Successfully.' }) 
        


@app.route('/close-position', methods=['POST'])
def close_position():
    profit = 0
    data = request.get_json()
    isClosePosition = False
    side = mt5.ORDER_TYPE_BUY if data['side'] == 'BUY' else mt5.ORDER_TYPE_SELL
    lot = data['lot']
    positions = mt5.positions_get(symbol=symbol)     
    if positions:
        last_position = positions[-1]

        if last_position.type == mt5.ORDER_TYPE_BUY and side == mt5.ORDER_TYPE_SELL:
            if last_position.profit < 0:
                profit = last_position.profit
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_SELL,
                "position": last_position.ticket,
                "price": mt5.symbol_info_tick(symbol).bid,
                #"deviation": 20,
                "magic": 202003,
                "comment": "",
                "type_time": mt5.ORDER_TIME_GTC,
                #"type_filling": mt5.ORDER_FILLING_IOC
            }
            mt5.order_send(close_request)
        if last_position.type == mt5.ORDER_TYPE_SELL and side == mt5.ORDER_TYPE_BUY:
            if last_position.profit < 0:
                profit = last_position.profit
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol, 
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY,
                "position": last_position.ticket,
                "price": mt5.symbol_info_tick(symbol).ask,
            # "deviation": 20,
                "magic": 202003,
                "comment": "",
                "type_time": mt5.ORDER_TIME_GTC,
                #"type_filling": mt5.ORDER_FILLING_IOC
            }
            mt5.order_send(close_request)

    return jsonify({'message': 'Close Position Successfully.', 'isClosePosition': isClosePosition, 'profit': profit }) 

@app.route('/update-position', methods=['POST'])
def updatePosition():
    data = request.get_json()
    position_id = int(data['position_id'])
    distance = 1
    position = mt5.positions_get(ticket=position_id)
    position = position[0]
    if not position:
        return jsonify({"error": "Position not found"}), 404
    
    price = position.price_open
    if position.type == mt5.ORDER_TYPE_BUY:
        sl_price = price - distance  # SL ở dưới giá Buy
        tp_price = price + distance  # TP ở trên giá Buy
    else:  # SELL
        sl_price = price + distance  # SL ở trên giá Sell
        tp_price = price - distance  # TP ở dưới giá Sell

    modify_request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": position_id,
        "sl": sl_price,
        "tp": tp_price,
    }

    mt5.order_send(modify_request)
    return jsonify({'message': 'Updated Successfully.' })       

@app.route('/trade', methods=['POST'])
def trade():
    data = request.get_json()
    lot = float(data['lot'])
    isClosePosition = False
    side = mt5.ORDER_TYPE_BUY if data['side'] == 'BUY' else mt5.ORDER_TYPE_SELL
    positions = mt5.positions_get(symbol=symbol)     
    tick = mt5.symbol_info_tick(symbol)
    distance = float(data.get('distance', 1))
    if side == mt5.ORDER_TYPE_BUY:
        price = tick.ask
        sl_price = price - distance  # SL ở dưới giá Buy
        tp_price = price + distance  # TP ở trên giá Buy
    else:  # SELL
        price = tick.bid
        sl_price = price + distance  # SL ở trên giá Sell
        tp_price = price - distance  # TP ở dưới giá Sell

    if positions is not None and len(positions) > 0:
        return jsonify({'message': 'Trade Successfully.', 'position_id': positions[0].ticket })
    request_trade = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": side, 
        # "sl": sl_price,          # Stop Loss
        # "tp": tp_price,   
        "magic": 202003,
        "comment": "",
        "type_time": mt5.ORDER_TIME_GTC,
        #"type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request_trade)
    position_id = result.order
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return jsonify({
            "error": "Trade failed",
            "details": result._asdict()
        }), 500

    return jsonify({'message': 'Trade Successfully.', 'isClosePosition': isClosePosition, 'position_id': position_id })
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    finally:
        mt5.shutdown()

