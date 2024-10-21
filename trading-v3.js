import ccxt from "ccxt";
import moment from "moment";
import dotenv from 'dotenv';
import crypto from 'crypto';
import axios from "axios";
import cron from 'node-cron'
import delay from "delay";

dotenv.config();
const API_KEY = process.env.API_KEY;
const SECRET_KEY = process.env.SECRET_KEY;
const BASE_URL = 'https://fapi.binance.com';
const SYMBOL = 'BTCUSDC';
let quantity = 0.004;
let leverage = 3;
let binance = new ccxt.binance({
    apiKey: API_KEY,
    secret: SECRET_KEY,
    'enableRateLimit': true,
    'options': {}
});

// Hàm tạo chữ ký
function getSignature(queryString, secret) {
    return crypto.createHmac('sha256', secret).update(queryString).digest('hex');
}
async function sendSignedRequest(method, urlPath, data) {
    const queryString = Object.keys(data).map(key => `${key}=${data[key]}`).join('&');
    const signature = getSignature(queryString, SECRET_KEY);
    const url = `${BASE_URL}${urlPath}?${queryString}&signature=${signature}`;
    try {
        let response = await axios({
            method: method,
            url: url,
            headers: { 'X-MBX-APIKEY': API_KEY }
        });

        return response.data;
    } catch (error) {
        console.error(error);
    }
}

async function setLeverage(symbol, leverage, timestamp) {
    await sendSignedRequest('POST', '/fapi/v1/leverage', { symbol, leverage, timestamp: Date.now() });
}

function getSide(bos_choch) {
    for(let i = bos_choch.data.length - 1; i >= 0; i--) {
        let bos_choch_item = bos_choch.data[i];
        if(bos_choch_item.fractals_high) {
            return 'SELL';
        } else if (bos_choch_item.fractals_low) {
            return 'BUY'
        }
    }
}

let incrementValue = 10000;
async function trading(count) {
    const url = `http://localhost:5001/klines`;
    const response = await axios.get(url);
    let prices = response.data;
    const bPrices = prices.map(price => ({
        open: price.open,
        high: price.high,
        low: price.low,
        close: price.close,
        volume: price.tick_volume,
        RSI: price.RSI,
        MACD: price.MACD,
        SMA: price.SMA,
        MACD_SIGNAL: price.MACD_SIGNAL,
        MACD_DIFF: price.MACD_DIFF,
        STOCH_K: price.STOCH_K,
        STOCH_D: price.STOCH_D,
        ROC_RATE: price.ROC_RATE,
        ATR: price.ATR
    }));
    let bPricesFinal = prices[prices.length - 2];
    bPrices.pop();

    let prediction = await axios.post('http://localhost:5001/predict', {
        Close: bPrices.map(item => +item.close - incrementValue),
        Open: bPrices.map(item => +item.open - incrementValue),
        Low: bPrices.map(item => +item.low - incrementValue),
        High: bPrices.map(item => +item.high - incrementValue),
        Volume: bPrices.map(item => +item.volume),
        RSI: bPrices.map(item => +item.RSI),
        MACD: bPrices.map(item => +item.MACD),
        MACD_SIGNAL: bPrices.map(item => +item.MACD_SIGNAL),
        MACD_DIFF: bPrices.map(item => +item.MACD_DIFF),
        SMA: bPrices.map(item => +item.SMA),
        STOCH_K: bPrices.map(item => +item.STOCH_K),
        STOCH_D: bPrices.map(item => +item.STOCH_D),
        ROC_RATE: bPrices.map(item => +item.ROC_RATE),
        ATR: bPrices.map(item => +item.ATR), 
    });

    let originPredictedPrice = prediction.data.predicted_next_closing_price;
    let originClosePrice = bPricesFinal.close;

    let predictedPrice = parseFloat(originPredictedPrice);
    let closePrice = parseFloat(originClosePrice);
    let side = 'SELL';
    if (originPredictedPrice + incrementValue > closePrice) {
        side = 'BUY'
    }
  
    if (count == 4) console.log(prediction.data.predicted_next_closing_price + incrementValue, closePrice)

    //let timestamp = Date.now()
    //setLeverage(SYMBOL, leverage, Date.now());
    const positions = await binance.fetchPositions(['BTCUSDC']) || [];
    if (positions.length) {
        let positionSide = positions[0].side == 'long' ? 'BUY' : 'SELL';
        if (positionSide == side) return;
        let closeSide = positionSide == 'SELL' ? 'BUY' : 'SELL';
 
        await sendSignedRequest('POST', '/fapi/v1/order', {
            symbol: SYMBOL,
            side: closeSide,
            type: 'MARKET',
            quantity: quantity,
            timestamp: Date.now()
        });
    }

    let obj = {
        symbol: SYMBOL,
        side,
        type: 'MARKET',
        quantity,
        timestamp: Date.now()
    }

    const data = sendSignedRequest('POST', '/fapi/v1/order', obj)



    return data;
}
async function runMultipleTimes(interval, times, action) {
    let count = 0;

    async function run() {
        if (count < times) {
            await action(count);
            count++;
            setTimeout(run, interval);
        }
    }

    await run();   
}    

await trading(4)
cron.schedule('*/15 * * * *', async () => {
    await runMultipleTimes(1000, 5, trading)
});
