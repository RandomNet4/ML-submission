const tf = require('@tensorflow/tfjs-node');

const loadModel = async () => {
    return await tf.loadGraphModel(`${process.env.MODEL_URL_STORAGE}`);
};

module.exports = loadModel;