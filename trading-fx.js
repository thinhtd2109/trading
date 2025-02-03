
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

    let predict = await axios.post(url, { start: 1 });

    let side = 'SELL';
    if (predict.data.predicted_next_closing_price == 0) {
        side = 'BUY'
    }

    // if (count_decrease == count_increase) {
    //     await axios.post(`http://localhost:5001/close-current-position`, { lot });
    //     return true;
    // };


    await axios.post(`http://localhost:5001/close-position`, {
        lot,
        side
    });
 
    let order = await axios.post(`http://localhost:5001/trade`, {
        lot,
        side,
    });


    // if (order.data.position_id) {
    //     await axios.post(`http://localhost:5001/update-position`, {
    //         position_id: order.data.position_id,
    //         side,
    //     });
    // }

   
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
cron.schedule('*/15 * * * *', async () => {
    await delay(19000);
    await tradingv2()
});     
              