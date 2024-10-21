import axios from "axios";
import ExcelJS from 'exceljs';
import fs from 'fs'
import ccxt from "ccxt";
import moment from "moment";
let binance = new ccxt.binance();
let SYMBOL = 'BTCUSDT';
// Hàm lấy dữ liệu từ API
async function fetchHistoricalData(symbol, interval, startTime, limit = 1000) {

    // const data = await binance.fetchOHLCV('BTC/USDC', interval, startTime, limit);
    // return data;
    const url = `https://fapi.binance.com/fapi/v1/klines`;
    const params = {
        symbol: SYMBOL,
        interval: interval,
        startTime: startTime,
        limit: limit
    };

    try {
        const response = await axios.get(url, { params });
        return response.data;
    } catch (error) {
        console.error('Error fetching data:', error);
        return null;
    }
}

// Hàm để lấy dữ liệu liên tục cho khoảng thời gian 10 năm
async function fetchFiveYearsData(symbol, interval) {
    const fiveYears = 10 * 365 * 24 * 60 * 60 * 1000; // 10 năm tính bằng milliseconds
    const endTime = new Date().getTime(); // Thời điểm hiện tại
    const startTime = moment('2019-09-14').valueOf();
    let currentTime = startTime;
    let allData = [];

    while (currentTime < endTime) {
        const data = await fetchHistoricalData(symbol, interval, currentTime);
        if (data && data.length > 0) {
            allData = allData.concat(data);
            const lastDataPoint = data[data.length - 1];
            currentTime = parseInt(lastDataPoint[6]); // Sử dụng thời gian kết thúc của candle cuối cùng làm thời gian bắt đầu cho lần lấy tiếp theo
        } else {
            break;
        }
    }

    return allData;
}

async function exportToCSV(data, filePath) {
    const workbook = new ExcelJS.Workbook(); // Tạo một workbook mới
    const sheet = workbook.addWorksheet('Historical Data'); // Thêm một worksheet

    // Thêm tiêu đề cột vào sheet
    sheet.columns = [
        { header: 'Open Time', key: 'openTime', width: 22 },
        { header: 'Open', key: 'open', width: 10 },
        { header: 'High', key: 'high', width: 10 },
        { header: 'Low', key: 'low', width: 10 },
        { header: 'Close', key: 'close', width: 10 },
        { header: 'Volume', key: 'volume', width: 15 },
        { header: 'Close Time', key: 'closeTime', width: 22 },
        { header: 'Quote Asset Volume', key: 'quoteAssetVolume', width: 18 },
        { header: 'Number of Trades', key: 'trades', width: 18 },
        { header: 'Taker Buy Base Asset Volume', key: 'takerBuyBase', width: 25 },
        { header: 'Taker Buy Quote Asset Volume', key: 'takerBuyQuote', width: 25 }
    ];

    // Thêm dữ liệu vào các hàng
    data.forEach(item => {
        sheet.addRow({
            openTime: new Date(item[0]).toISOString(),
            open: item[1],
            high: item[2],
            low: item[3],
            close: item[4],
            volume: item[5],
            closeTime: new Date(item[6]).toISOString(),
            quoteAssetVolume: item[7],
            trades: item[8],
            takerBuyBase: item[9],
            takerBuyQuote: item[10]
        });
    });

    // Lưu dữ liệu vào tệp CSV
    const csv = await workbook.csv.writeBuffer();
    fs.writeFileSync(filePath, csv);
    console.log(`File is written to ${filePath}`);
}

// Use the function to fetch data and export to CSV
fetchFiveYearsData(SYMBOL, '15m').then(data => {
    exportToCSV(data, `./${SYMBOL}_Historical_Data.csv`);
});
