import MetaTrader5 as mt5
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import pickle
import ta
import ta.momentum
import ta.trend
import ta.volatility
from tensorflow.keras.models import load_model
from datetime import datetime
from smartmoneyconcepts import smc
from tapy import Indicators
import pytz

app = Flask(__name__)
# Khởi tạo MetaTrader 5
if not mt5.initialize():
    print("Initialization failed")
    mt5.shutdown()
    exit()

# login = 148094627
# password = "Ducthinh@2109" 
# server = "Exness-MT5Real18"

login = 181130580  
password = "Ducthinh@2109"
server = 'Exness-MT5Trial6' 

symbol = "BTCUSD"  
isReverse = False
      
# Load the model and scalers
model = load_model('./model/predict_model_BTC_3.h5')

if not mt5.login(login=login, password=password, server=server):
    print("Login failed")
    mt5.shutdown()
    exit()
 
def load_scalers():
    with open('./model/predict_scalers_BTC_3.pkl', 'rb') as f:
        scalers = pickle.load(f) 
    return scalers

 
def preprocess_input(data, scalers):
    """ Preprocess the input JSON data to a format suitable for the model """
    time_step = 10
    features = ["body"]

    num_features = len(features)
    input_data = np.zeros((1, time_step, num_features))
    
    # Apply individual scaling and format to input shape
    for i, feature in enumerate(features):
        values = data[feature]
        scaled_values = scalers['Price'].transform(np.array(values).reshape(-1, 1)).flatten()
        input_data[0, :, i] = scaled_values
    
    return input_data.reshape(input_data.shape[1], input_data.shape[2])

@app.route('/predict', methods=['POST'])
def predict_next_price():
    # Expecting JSON input with 60-timestep data for each feature  
    json_data = request.get_json()
    scalers = load_scalers()
    # Preprocess the input to match the model's input shape
    input_data = preprocess_input(json_data, scalers)

    print(input_data)

    # Predict the next closing price
    next_closing_price = model.predict(input_data)
    predicted_price = scalers['Price'].inverse_transform(next_closing_price)

    # Return the prediction in JSON format
    return jsonify({'predicted_next_closing_price': float(predicted_price[0, 0])})

@app.route('/histories', methods=['GET'])
def get_histories():
    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime(2020, 1, 10, tzinfo=timezone)
    rates = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_M15, utc_from, 50000)
    rates_frame = pd.DataFrame(rates)
    RSI = ta.momentum.RSIIndicator(rates_frame['close'])
    MACD = ta.trend.MACD(rates_frame['close'])
    STOCH = ta.momentum.StochasticOscillator(rates_frame['high'], rates_frame['low'], rates_frame['close'])
    ROC = ta.momentum.ROCIndicator(close=rates_frame['close'], window=12)
    ATR = ta.volatility.AverageTrueRange(high=rates_frame['high'], low=rates_frame['low'], close=rates_frame['close'], window=14)
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame['RSI'] = RSI.rsi()
    rates_frame['MACD'] = MACD.macd()
    rates_frame['STOCH_K'] = STOCH.stoch()
    rates_frame['STOCH_D'] = STOCH.stoch_signal()
    rates_frame['ROC_RATE'] = ROC.roc() 
    rates_frame['ATR'] = ATR.average_true_range()
    rates_json = rates_frame.to_dict(orient='records')


    # Return the prediction in JSON format
    return jsonify(rates_json) 
 
@app.route('/klines', methods=['GET'])
def get_klines():
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    rates_frame = pd.DataFrame(rates)
    RSI = ta.momentum.RSIIndicator(rates_frame['close'])
    MACD = ta.trend.MACD(rates_frame['close'])
    STOCH = ta.momentum.StochasticOscillator(rates_frame['high'], rates_frame['low'], rates_frame['close'])
    ROC = ta.momentum.ROCIndicator(close=rates_frame['close'], window=12)
    ATR = ta.volatility.AverageTrueRange(high=rates_frame['high'], low=rates_frame['low'], close=rates_frame['close'], window=14)
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame['RSI'] = RSI.rsi()
    rates_frame['MACD'] = MACD.macd()
    rates_frame['STOCH_K'] = STOCH.stoch()
    rates_frame['STOCH_D'] = STOCH.stoch_signal()
    rates_frame['ROC_RATE'] = ROC.roc()
    rates_frame['ATR'] = ATR.average_true_range()
    rates_json = rates_frame.to_dict(orient='records')


    # Return the prediction in JSON format
    return jsonify(rates_json[-11:]) 

@app.route('/positions', methods=['GET'])
def get_positions():
    positions = mt5.positions_get(symbol=symbol)     
    df=pd.DataFrame(list(positions),columns=positions[0]._asdict().keys())
    json = df.to_dict(orient='records')
    return jsonify(json[-1])

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
        elif last_position.type == mt5.ORDER_TYPE_SELL and side == mt5.ORDER_TYPE_BUY:
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
        profit = last_position.profit
    return jsonify({'message': 'Close Position Successfully.', 'profit': profit }) 
        

@app.route('/trade', methods=['POST'])
def trade():
    data = request.get_json()
    lot = float(data['lot'])
    isClosePosition = False
    side = mt5.ORDER_TYPE_BUY if data['side'] == 'BUY' else mt5.ORDER_TYPE_SELL
    positions = mt5.positions_get(symbol=symbol)     
    if positions:
        return jsonify({'message': 'Trade Successfully.', 'isClosePosition': isClosePosition })
    request_trade = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": side,
        "magic": 202003,
        "comment": "",
        "type_time": mt5.ORDER_TIME_GTC,
        #"type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request_trade)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return jsonify({
            "error": "Trade failed",
            "details": result._asdict()
        }), 500

    return jsonify({'message': 'Trade Successfully.', 'isClosePosition': isClosePosition })
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5001, debug=True)
    finally:
        mt5.shutdown()

