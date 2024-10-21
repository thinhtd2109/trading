import MetaTrader5 as mt5
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import pickle
import ta
import ta.momentum
import ta.trend
import ta.volatility
from joblib import dump, load
import joblib


from dtaidistance import dtw

def dtw_distance(x, y):
    return dtw.distance(x, y) 

app = Flask(__name__)
# Khởi tạo MetaTrader 5
if not mt5.initialize():
    print("Initialization failed")
    mt5.shutdown()
    exit()

login = 103865038
password = "Ducthinh@2109" 
server = "Exness-MT5Real15" 

# login = 181130580  
# password = "Ducthinh@2109"
# server = 'Exness-MT5Trial6'

symbol = "XAUUSD"  
isReverse = False 
   
# Load the model and scalers

model = load('logistic_model.joblib')

with open('./scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

if not mt5.login(login=login, password=password, server=server):
    print("Login failed")
    mt5.shutdown() 
    exit() 
 

@app.route('/predict', methods=['GET'])
def predict_next_price():
    feature_names = ['RSI', 'STOCH_K', 'STOCH_D', 'ROC_RATE', 'ATR']
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, 50)
    new_data = pd.DataFrame(rates)
    
    RSI = ta.momentum.RSIIndicator(new_data['close'], fillna=True, window=5)
    STOCH = ta.momentum.StochasticOscillator(new_data['high'], new_data['low'], new_data['close'], window=5, fillna=True)
    ROC = ta.momentum.ROCIndicator(close=new_data['close'], window=5, fillna=True)
    ATR = ta.volatility.AverageTrueRange(high=new_data['high'], low=new_data['low'], close=new_data['close'], window=5, fillna=True)
    
    new_data['RSI'] = RSI.rsi()
    new_data['ATR'] = ATR.average_true_range()
    new_data['STOCH_K'] = STOCH.stoch()
    new_data['STOCH_D'] = STOCH.stoch_signal()
    new_data['ROC_RATE'] = ROC.roc()

    # Chọn các đặc trưng để dự đoán
    features = new_data[feature_names].iloc[-1].values.reshape(1, -1)
    new_data_df = pd.DataFrame(features, columns=feature_names)
    print(new_data_df)
    features_scaled = scaler.transform(new_data_df)

    print(features_scaled)
    
    # Dự đoán giá đóng cửa tiếp theo
    prediction = model.predict(features_scaled)
    

    # Trả kết quả dự đoán
    is_buy = int(prediction[0])
    return jsonify({'predicted_next_closing_price': is_buy})
    

@app.route('/positions', methods=['GET'])
def get_positions():
    positions = mt5.positions_get(symbol='EURUSDm')     
    if not positions:
        return jsonify({ 'data': None })
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
            if last_position.profit < 0:
                isClosePosition = True
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
                isClosePosition = True
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

