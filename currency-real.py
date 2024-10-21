import MetaTrader5 as mt5
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import pickle
import ta
import ta.momentum
import ta.trend
import ta.volatility
from tapy import Indicators
from tensorflow.keras.models import load_model # type: ignore # ignore
import warnings

app = Flask(__name__)
# Khởi tạo MetaTrader 5
if not mt5.initialize():
    print("Initialization failed")
    mt5.shutdown()
    exit()

login = 103865038
password = "Ducthinh@2109" 
server = "Exness-MT5Real15"

# login = 116761935
# password = "Ducthinh@2109"
# server = 'Exness-MT5Trial6'  

symbol = "XAUUSD"  
isReverse = False 
  
# Load the model and scalers
model = load_model('./model/predict_model_2.h5')

if not mt5.login(login=login, password=password, server=server):
    print("Login failed")
    mt5.shutdown()
    exit()
 
def load_scalers(): 
    with open('./model/scalers_1.pkl', 'rb') as f:
        scalers = pickle.load(f)
    return scalers

 
def preprocess_input(data, scalers):
    """ Preprocess the input JSON data to a format suitable for the model """
    time_step = 60
    features = ["Close", "High", "Low", "Open"]
    num_features = len(features)
    input_data = np.zeros((1, time_step, num_features))
    
    # Apply individual scaling and format to input shape
    for i, feature in enumerate(features):
        values = data[feature]
        scaled_values = scalers['Price'].transform(np.array(values).reshape(-1, 1)).flatten()
        input_data[0, :, i] = scaled_values
    
    return input_data

def calculate_parabolic_sar(df):
    warnings.filterwarnings("ignore")
    # Assumed/Standard Values
    initial_af = 0.1
    max_af = 0.2
    af_increment = 0.1

    # Create columns for SAR, AF, and EP
    df['SAR'] = 0.0 
    df['AF'] = 0.0
    df['EP'] = 0.0

    # Determine the starting trend (True for uptrend, False for downtrend)
    uptrend = df['close'].iloc[0] < df['close'].iloc[1] 

    # Initialize first row values
    df['SAR'].iloc[0] = df['low'].iloc[0] if uptrend else df['high'].iloc[0]
    df['AF'].iloc[0] = initial_af
    df['EP'].iloc[0] = df['high'].iloc[0] if uptrend else df['low'].iloc[0]

    for i in range(1, len(df)):
        prev_sar = df['SAR'].iloc[i - 1]
        prev_af = df['AF'].iloc[i - 1]
        prev_ep = df['EP'].iloc[i - 1]

        # Calculate SAR for today
        df['SAR'].iloc[i] = prev_sar + prev_af * (prev_ep - prev_sar)

        # Check for trend reversal
        if (uptrend and df['SAR'].iloc[i] > df['low'].iloc[i]) or (not uptrend and df['SAR'].iloc[i] < df['high'].iloc[i]):
            uptrend = not uptrend  # Reverse the trend
            df['SAR'].iloc[i] = prev_ep  # Set SAR to the EP of the previous trend
            df['AF'].iloc[i] = initial_af  # Reset AF
            df['EP'].iloc[i] = df['high'].iloc[i] if uptrend else df['low'].iloc[i]  # Set EP to today's high/low
        else:
            # Update AF and EP if a new high/low is made
            if (uptrend and df['high'].iloc[i] > prev_ep) or (not uptrend and df['low'].iloc[i] < prev_ep):
                df['AF'].iloc[i] = min(max_af, prev_af + af_increment)  # Increment AF, cap at max_af
                df['EP'].iloc[i] = df['high'].iloc[i] if uptrend else df['low'].iloc[i]  # Update EP to today's high/low
            else:
                df['AF'].iloc[i] = prev_af  # Carry over AF
                df['EP'].iloc[i] = prev_ep  # Carry over EP

    return df


@app.route('/test', methods=['GET'])
def test():
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 200)
    rates_frame = pd.DataFrame(rates)
    rates_frame = calculate_parabolic_sar(rates_frame)
    rates_frame['down_trend'] = rates_frame['EP'] <= rates_frame['close']
    rates_frame['up_trend'] = rates_frame['EP'] > rates_frame['close']
    return jsonify(rates_frame.to_dict(orient='records'))

@app.route('/predict', methods=['POST'])
def predict_next_price():
    # Expecting JSON input with 60-timestep data for each feature  
    json_data = request.get_json()

    scalers = load_scalers() 
    # Preprocess the input to match the model's input shape
    input_data = preprocess_input(json_data, scalers)

    # Predict the next closing price
    next_closing_price = model.predict(input_data)
    predicted_price = scalers['Price'].inverse_transform(next_closing_price)

    # Return the prediction in JSON format
    return jsonify({'predicted_next_closing_price': float(predicted_price[0, 0])})


@app.route('/sar', methods=['GET'])
def sar():
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 200)
    rates_frame = pd.DataFrame(rates)
    PSAR = ta.trend.PSARIndicator(close=rates_frame['close'], high=rates_frame['high'], low=rates_frame['low'], max_step=0.25)
    RSI = ta.momentum.RSIIndicator(close=rates_frame['close'])
    MACD = ta.trend.MACD(close=rates_frame['close'])
    rates_frame['MACD'] = MACD.macd()
    rates_frame['Signal_Line'] = MACD.macd_signal()
    rates_frame['RSI'] = RSI.rsi()
    rates_frame['up_trend'] = PSAR.psar_up().fillna(False)
    rates_frame['down_trend'] = PSAR.psar_down().fillna(False)
    return jsonify(rates_frame.to_dict(orient='records')[-30:])


@app.route('/klines', methods=['GET'])
def get_klines():
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 200)
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
    return jsonify(rates_json[-61:]) 

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

