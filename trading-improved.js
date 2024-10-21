import fs from 'fs'
import axios from 'axios';
import moment from 'moment';
// Đọc file CSV lịch sử Forex
const data = fs.readFileSync('./data-test/OANDA_EURUSD_TEST_Updated.csv', 'utf8');
const rows = data.split('\n');
const prices = [];

for (let i = 1; i < rows.length; i++) {
    const row = rows[i].split(',');
    const previousRow = rows[i - 1].split(',');
    // RSI,MACD,MFI,ADX_AVG
    const price = {
        open: row[1],
        high: row[2],
        low: row[3],
        close: row[4],
        volume: row[5],
        RSI: row[6],
        MACD: row[7],
        STOCH_K: row[8],
        STOCH_D: row[9],
        ROC_RATE: row[10],
        ATR: row[11],
    };
    // const price = {
    //     open: parseFloat(row[1]),
    //     high: parseFloat(row[2]),
    //     low: parseFloat(row[3]),
    //     close: parseFloat(row[4]),
    //     volume: parseFloat(row[5]),
    //     rsi: parseFloat(row[6]),
    //     macd: parseFloat(row[7]),
    //     sma: parseFloat(row[8]),
    //     ema: parseFloat(row[9]),
    //     H_BOL: parseFloat(row[10]),
    //     L_BOL: parseFloat(row[11]),
    //     V_BOL: parseFloat(row[12]),
    //     Target: parseFloat(row[4]) - parseFloat(previousRow[4]) >= 0 ? 2 : 1,
    //     time: row[0]
    // };
    prices.push(price);
}
// ['Close', 'Open', 'High', 'Low', 'SMA', 'EMA', 'H_BOL', 'L_BOL', 'V_BOL']
const predictNextClosingPrice = async (history) => {

    const response = await axios.post('http://localhost:5000/predict', {

        Close: history.map(item => +item.close),
        Low: history.map(item => +item.low),
        High: history.map(item => +item.high),
        Volume: history.map(item => +item.volume),
        Open: history.map(item => +item.open),
        RSI: history.map(item => +item.RSI),
        MACD: history.map(item => +item.MACD),
        STOCH_K: history.map(item => +item.STOCH_K),
        STOCH_D: history.map(item => +item.STOCH_D),
        ROC_RATE: history.map(item => +item.ROC_RATE),
        
        // ATR: history.map(item => +item.atr),
    });

    return parseFloat(response.data.predicted_next_closing_price)
};


// Backtest mô hình dự đoán
let profit = 0;
let maxProfit = -999999999;
let maxLoss = 999999999999;
let cash = 0;

let winCount = 0;
let loseCount = 0;

let string = '';



let count = 0;

let histories = '';


function splitArray(history) {
    let timeSteps = 1;

    let chunks = [];
    for (let i = 0; i < history.length; i += timeSteps) {
        chunks.push(history.slice(i, i + timeSteps));
    }
    return chunks;
}

let time_steps = 32;
for (let i = time_steps; i < prices.length; i++) {
    const history = prices.slice(i - time_steps, i) // Lấy 60 dữ liệu lịch sử gần nhất

    const currentCandle = prices[i];
    const predictedPrice = await predictNextClosingPrice(history);

    const bPricesFinal = history[history.length - 1];
    const currentOpen = parseFloat(currentCandle.open);
    const currentClose = parseFloat(currentCandle.close);

    let side = 'SELL';
    // if (predictedPrice > parseFloat(bPricesFinal.close)) {
    //     side = 'BUY'
    // }
    if (predictedPrice > 0.5) {
        side = 'BUY'
    }


    let distance = Math.abs(currentClose - currentOpen);
    if (side == 'BUY') {
        if (currentClose > currentOpen) {
            cash += distance;
            profit = distance;
            winCount++;
        } else {
            cash -= distance;
            profit = -distance;
            loseCount++;
        }
    } else {
        if (currentClose > currentOpen) {
            cash -= distance;
            profit = -distance;
            loseCount++;
        } else {
            winCount++;
            cash += distance;
            profit = distance;
        }
    }
    console.log(winCount, loseCount, side, predictedPrice)
    if (maxLoss > cash) maxLoss = cash;
    if (maxProfit < cash) maxProfit = cash;
    count++;
    string += `Total Profit ${moment(currentCandle['time']).format('DD-MM-YYYY HH:mm')} ${profit} PIP || Total Balance: ${cash} PIP \n`

}

console.log('Tỉ lệ thắng ' + ((winCount) / (loseCount + winCount)) * 100)

fs.writeFile('output.txt', string, (err) => {
    if (err) {
        console.error('Ghi vào tệp bị lỗi:', err);
        return;
    }
});


console.log(`Lợi nhuận tổng: ${profit}, Lợi nhuận tối đa: ${maxProfit}, Lỗ tối đa: ${maxLoss}, lời tối đa: ${maxProfit}`);
  