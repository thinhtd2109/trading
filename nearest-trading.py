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
import trading_utils 
app = Flask(__name__) 
# Khởi tạo MetaTrader 5 
if not mt5.initialize():
    print("Initialization failed")
    mt5.shutdown()
    exit()

# login = 103865038
# password = "Ducthinh@2109" 
# server = "Exness-MT5Real15" 

login = 79364337  
password = "Ducthinh@2109"
server = 'Exness-MT5Trial8'

symbol = "XAUUSD"   
isReverse = False  

  
if not mt5.login(login=login, password=password, server=server):  
    print("Login failed")
    mt5.shutdown() 
    exit() 
 
# # List of indicators
# indicators = ['AO', 'TSI', 'RSI', 'PPO', 'STOCH_RSI', 'STOCH', 'UO', 'WR', 'ROC']

# # Dictionary to store the models
# models = {}

# # Loop to load models dynamically
# for indicator in indicators:
#     model_name = f'nearest_neighbors_model_XAUUSD_{indicator}.joblib'
#     models[indicator] = load(model_name)

@app.route('/predict', methods=['POST'])
def predict_next_price():
    # Fetch historical rates
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)
    new_data = pd.DataFrame(rates)
    new_data['body'] = new_data['close'] - new_data['open']
    
    # new_data = trading_utils.processing_data(new_data=new_data)

    # features = []

    # lags = range(0, 10)

    # stoch = [f'STOCH_{i}' for i in lags]
    # stoch_rsi = [f'STOCH_RSI_{i}' for i in lags]
    # tsi = [f'TSI_{i}' for i in lags]
    # rsi = [f'RSI_{i}' for i in lags]

    # features += stoch 
    # features += tsi
    # features += rsi 
    # features += stoch_rsi 
     

    latest_candle = new_data.iloc[-1]

    # input_data = pd.DataFrame([latest_candle], columns=features)

    # # signal = trading_utils.predict_nearest(window_size=1, new_data=new_data, df_historical=df_historical, model=model)
    # y_pred = rf_classifier.predict(input_data)
    # # Extracting the predicted value and converting it to an integer
    # predicted_value = int(y_pred[0])  # Extract the first element from the prediction

    return jsonify({'predicted_next_closing_price': 1, 'close': latest_candle['close'], 'volume': latest_candle['tick_volume'] })


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
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)
    rates_frame = pd.DataFrame(rates)
    # rates_frame['body'] = rates_frame['close'] - rates_frame['open']
    # rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s') + timedelta(hours=7)

    # Định dạng lại thời gian theo ISO 8601 với múi giờ UTC+7
    # rates_frame['time'] = rates_frame['time'].dt.strftime("%Y-%m-%dT%H:%M:%S+07:00")
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

