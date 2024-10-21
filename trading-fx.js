
import dotenv from 'dotenv';
import axios from "axios";
import cron from 'node-cron'
import delay from "delay";
dotenv.config();

const url = `http://localhost:5001/get-open`;

let current_close = 0;
async function tradingv2(count, times) {
    let lot = 0.01
    const url = `http://localhost:5001/predict`;

    // let predict = await axios.post(url, { start: 1 });
    let multiplier = 10
    let closePrices = [];
    let time = 1
    // Collect 5 close prices at 1-second intervals
    for (let i = 0; i < (time * 60 * multiplier); i++) {
        let predict = await axios.post(url, { start: 1 });

        closePrices.push(predict.data.close)

        // Wait for 1 second before the next iteration
        await new Promise(resolve => setTimeout(resolve, 1000 / multiplier));
    }
    let count_increase = 0;
    let count_decrease = 0;
    let trend = 'neutral'; // Biến để theo dõi xu hướng hiện tại

    for (let i = 0; i < closePrices.length - 1; i++) {
        if (closePrices[i + 1] > closePrices[i]) {

            if (trend !== 'increasing') {
                count_increase += 1;
                trend = 'increasing';
            }
        } else if (closePrices[i + 1] < closePrices[i]) {

            if (trend !== 'decreasing') {
                count_decrease += 1;
                trend = 'decreasing';
            }
        } else {
            trend = 'neutral'; // Nếu giá không đổi, đặt lại xu hướng
        }
    }
    console.log(count_increase, count_decrease)
    let side = 'SELL';
    if (count_increase < count_decrease) {
        side = 'BUY';
    }

    if (count_decrease == count_increase) {
        await axios.post(`http://localhost:5001/close-current-position`, { lot });
        return true;
    };


    await axios.post(`http://localhost:5001/close-position`, {
        lot,
        side
    });

    await axios.post(`http://localhost:5001/trade`, {
        lot,
        side,
    });
    return true;
    //return bos_choch.data      

}

async function runMultipleTimes(interval, times, action) {
    let count = 0;

    async function run() {
        if (count < times) {
            let isEnd = await action(count, times);
            count++;
            setTimeout(run, interval);
        }
    }

    await run();
}

//await tradingv2()
cron.schedule('4,9,14,19,24,29,34,39,44,49,54,59 * * * *', async () => {
    await tradingv2()
});    
