
import dotenv from 'dotenv';
import axios from "axios";
import cron from 'node-cron'
import delay from "delay";
dotenv.config();

const PREDICT_URI = process.env.ENV == 'LOCAL' ? 'http://localhost:5001' : process.env.PREDICT_URI;

async function tradingv2(count, times) {
    let lot = 0.01
    const url = `${PREDICT_URI}/predict`;

    let predict = await axios.post(url, { start: 1 });

    let side = 'SELL';
    if (predict.data.predicted_next_closing_price == 1) {
        side = 'BUY'
    }

    await axios.post(`${PREDICT_URI}/close-position`, {
        lot,
        side
    });
 
    let order = await axios.post(`${PREDICT_URI}/trade`, {
        lot,
        side,
    });


   
    return true; 

}
await tradingv2()
cron.schedule('*/15 * * * *', async () => {
    await delay(25000);
    await tradingv2()
});     
    