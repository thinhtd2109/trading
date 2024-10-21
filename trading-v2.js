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
let quantity = 0.003;
let leverage = 3;
let binance = new ccxt.binance({
    apiKey: API_KEY,
    secret: SECRET_KEY,
    'enableRateLimit': true,
    'options': { 
        'defaultType': 'future' 
    }
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

async function trading() {
    const prices = await binance.fetchOHLCV('BTC/USDT', '15m', undefined, 10)
    // const prices = await sendSignedRequest('GET', `/fapi/v1/klines`, {
    //     limit: 10,
    //     symbol: SYMBOL,
    //     interval: '15m',
    // })

    // const url = `https://api.binance.com/api/v3/klines`;
    // const params = {
    //     symbol: 'BTCUSDC',
    //     interval: '15m',
    //     limit: 61
    // };
    // const response = await axios.get(url, { params });
    // let prices = response.data;
    const bPrices = prices.map(price => ({
        timestamp: moment(price[0]).format('YYYY-MM-DD HH:mm'),
        open: price[1],
        high: price[2],
        low: price[3],
        close: price[4],
        volume: price[5],
        isIncrease: price[4] > price[1] ? true : false
    }));
    console.log(bPrices)
}

await trading()
