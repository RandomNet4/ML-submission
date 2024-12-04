const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
async function binClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();
        const prediction = model.predict(tensor);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;
        let label, suggestion;
        if (confidenceScore > 50) {
            label = 'Cancer';
            suggestion = 'jangan panik. Segera periksa ke dokter untuk mendapatkan penanganan lebih lanjut..';
        } else {
            label = 'Non-cancer';
            suggestion = 'Jaga kesehatan dan pola hidup sehat..';
        }
        return { label, suggestion };
    } catch (error) {
        throw new InputError('Terjadi kesalahan dalam melakukan prediksi');
    }
}
module.exports = binClassification;