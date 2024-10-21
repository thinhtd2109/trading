import ccxt from "ccxt";
import moment from "moment";
import dotenv from 'dotenv';
import crypto from 'crypto';
import axios from "axios";
import cron from 'node-cron';
import delay from 'delay';

import { RSI, MACD } from 'technicalindicators';


dotenv.config();
const API_KEY = process.env.API_KEY;
const SECRET_KEY = process.env.SECRET_KEY;
const BASE_URL = 'https://fapi.binance.com';
const SYMBOL = 'BTCUSDC';
let quantity = 0.003;
let leverage = 2;
let binance = new ccxt.binance();

async function executeTrade(exchange, symbol, amount, leverage, side, takeProfitPrice, stopLossPrice) {
    await exchange.setLeverage(leverage, symbol);

    const buyOrder = await exchange.createMarketOrder(symbol, side.toLowerCase(), amount);
    console.log('Buy order executed:', buyOrder);

    const oppositeSide = side === 'BUY' ? 'sell' : 'buy';

    await exchange.createOrder(symbol, 'limit', oppositeSide, amount, takeProfitPrice, {
        reduceOnly: true
    });
    console.log('Take Profit set at:', takeProfitPrice);

    await exchange.createOrder(symbol, 'stopMarket', oppositeSide, amount, undefined, {
        stopPrice: stopLossPrice,
        reduceOnly: true
    });
    console.log('Stop Loss set at:', stopLossPrice);
}

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

async function setLeverage(symbol, leverage) {
    await sendSignedRequest('POST', '/fapi/v1/leverage', { symbol, leverage, timestamp: Date.now() });
}

async function trading() {
    const prices = await sendSignedRequest('GET', `/fapi/v1/klines`, {
        limit: 20,
        symbol: SYMBOL,
        interval: '15m',
    })

    const bPrices = prices.map(price => ({
        timestamp: moment(price[0]).format('YYYY-MM-DD HH:mm'),
        open: price[1],
        high: price[2],
        low: price[3],
        close: price[4],
        volume: price[5],
        isIncrease: price[4] > price[1] ? true : false
    }));

    let bPricesFinal = bPrices[bPrices.length - 2];
    const closes = prices.map(item => item[4]);
    const rsiValues = RSI.calculate({ values: closes, period: 3 });

    const peaks = [];
    for (let i = 1; i < rsiValues.length - 1; i++) {
        if (rsiValues[i] > rsiValues[i - 1] && rsiValues[i] > rsiValues[i + 1]) {
            peaks.push({ index: i, value: rsiValues[i] });
        }
    }


    peaks.sort((a, b) => b.value - a.value);


    let side = 'SELL';
    if (peaks.length >= 2) {
        const highestPeaks = peaks.slice(0, 2);  // Lấy hai đỉnh cao nhất
        let lastPeak = highestPeaks[0].index > highestPeaks[1].index ? highestPeaks[0] : highestPeaks[1];
        let secondLastPeak = highestPeaks[0].index < highestPeaks[1].index ? highestPeaks[0] : highestPeaks[1];
        if (lastPeak.value > secondLastPeak.value) {
            side = 'BUY';
        }
    } else {
        console.log("Không đủ đỉnh để phân tích.");
    }

    console.log(side)

    // let side = 'SELL';
    // if (bPricesFinal.close > bPricesFinal.open) {
    //     side = 'BUY';
    // }

    // ---- rervese
    // let side = 'BUY';
    // if (bPricesFinal.close > bPricesFinal.open) {
    //     side = 'SELL';
    // }
    setLeverage(SYMBOL, leverage);
    const timestamp = await axios.get('https://api.binance.com/api/v3/time', {
        headers: { 'X-MBX-APIKEY': API_KEY }
    })

    let list = await sendSignedRequest('GET', '/fapi/v1/allOrders', { timestamp: timestamp.data.serverTime, limit: 2, symbol: SYMBOL });
    let finalItem = list[list.length - 1];

    if (!finalItem.reduceOnly && finalItem.type == 'MARKET') {
        let closeSide = finalItem.side == 'SELL' ? 'BUY' : 'SELL';

        const timestamp = await axios.get('https://api.binance.com/api/v3/time', {
            headers: { 'X-MBX-APIKEY': API_KEY }
        })

        if (finalItem.side === side) return;

        await sendSignedRequest('POST', '/fapi/v1/order', {
            symbol: SYMBOL,
            side: closeSide,
            type: 'MARKET',
            quantity: quantity,
            timestamp: timestamp.data.serverTime
        });
    }

    let obj = {
        symbol: SYMBOL,
        side,
        type: 'MARKET',
        quantity,
        timestamp: timestamp.data.serverTime
    }

    const data = sendSignedRequest('POST', '/fapi/v1/order', obj)
    return data;
}


//await trading()
cron.schedule('*/15 * * * *', async () => {
    await delay(10 * 100)
    await trading();
});
