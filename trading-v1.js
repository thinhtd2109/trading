import ccxt from "ccxt";
import moment from "moment";
import { Parser } from 'json2csv';
import fs from 'fs'
import axios from "axios";
import csvParser from "csv-parser";

const apiKey = "ML53CjFfxEjoTCs6Cs8ltPhSI2mUbvbuAC6BJ4dF68ljWBmQbtcluJEdC8ANahhH";
const secretKey = "vS3HrZDdaIYDk0sqBYv5Jus7aZw6LTGiX2xau6vu9dsIamTbM5TUdT5gkQLZGlQk";

const binance = new ccxt.binance({ apiKey, secret: secretKey });
binance.setSandboxMode(true);
let min = 999999999;
const limit = 1000; // Số lượng nến cần tải (giới hạn là 1000)
async function main() {
    let bPrices = [];
    let string = '';
    let input = [];
    fs.createReadStream('BTCUSDT_Historical_Data.csv')
        .pipe(csvParser())
        .on('data', (data) => {
            bPrices.push(data);
        })
        .on('end', async () => {
            let cash = 0;

            let dailyDay = 0;
            let winCount = 0;
            let lossCount = 0;
            let min = 9999999999;
            for (let i = 1; i < bPrices.length; i++) {
                let profit = 0;
                const currentCandle = bPrices[i];
                const previousCandle = bPrices[i - 1];
                const currentOpen = +currentCandle.Open;
                const currentClose = +currentCandle.Close;

                const previousOpen = +previousCandle.Open;
                const previousLow = +previousCandle.Low;
                const previousHigh = +previousCandle.High;
                const previousClose = +previousCandle.Close;
                const previousVolume = +previousCandle.Volume;

                input.push([previousOpen, previousClose])

                //let side = 'SELL';
                // if (+prediction.data.prediction[0] > +bPricesFinal.close) {
                //     side = 'BUY'
                // }

                // let distance = Math.abs(currentClose - currentOpen) - 0.06



                // if (side == 'BUY') {
                //     if (currentClose > currentOpen) {
                //         cash += distance;
                //         profit = +distance;
                //         dailyDay += distance;
                //         winCount++;
                //     } else {
                //         lossCount++;
                //         cash -= distance;
                //         profit = -distance
                //         dailyDay -= distance
                //     }
                // }

                // if (profit > 0) string += `Total Profit: ${profit} USDT || Total Balance ${moment(currentCandle['Date']).format('DD-MM-YYYY HH:mm')} => ${cash} USDT \n`;

            }

            let prediction = await axios.post('http://localhost:5000/predict', {
                input
            });
            for (let i = 1; i < bPrices.length; i++) {
                let profit = 0;
                const previousCandle = bPrices[i - 1]
                const currentCandle = bPrices[i];
                const currentOpen = +currentCandle.Open;
                const currentClose = +currentCandle.Close;
                let side = 'SELL';
                if (parseFloat(prediction.data.prediction[i]) > parseFloat(previousCandle.Close)) {
                    side = 'BUY'
                };


                let distance = Math.abs(currentClose - currentOpen) - 0.06
                if (side == 'BUY') {
                    if (currentClose > currentOpen) {
                        cash += distance;
                        profit = distance;
                        winCount++;
                    } else {
                        lossCount++;
                        cash -= distance;
                        profit = -distance
                    }
                } else {
                    if (currentClose > currentOpen) {
                        cash -= distance;
                        profit = -distance;
                        lossCount++;
                    } else {
                        cash += distance;
                        profit = distance;
                        winCount++;
                    }
                }

                if (min > profit) min = profit



                // if (profit == 0) {
                //     console.log("profit", profit, 'side', side, 'distance', distance)
                //     console.log(currentClose, currentOpen)
                //     return;
                // }

                if (moment(currentCandle['Open Time']).format('DD-MM-YYYY HH:mm') == '29-04-2024 15:00') {
                    console.log(parseFloat(prediction.data.prediction[i]) > parseFloat(previousCandle.Close))
                    console.log(side)
                }
                string += `Total Profit ${moment(currentCandle['Open Time']).format('DD-MM-YYYY HH:mm')} ${profit} USDT || Total Balance: ${cash} USDT \n`;

            }
            console.log(lossCount)
            fs.writeFile('output.txt', string, (err) => {
                if (err) {
                    console.error('Ghi vào tệp bị lỗi:', err);
                    return;
                }
            });
        });





}


main()