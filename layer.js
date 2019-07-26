const model = tf.sequential();

const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'sigmoid'
});

const output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
});

model.add(hidden);
model.add(output);

const sgdOpt = tf.train.sgd(0.1);

model.compile({
    optimizer: sgdOpt,
    loss: tf.losses.meanSquaredError
})


const xs = tf.tensor2d([
    [0, 0],
    [0.5, 0.5],
    [1, 1],
]);

const ys = tf.tensor2d([
    [1],
    [0.5],
    [0]
]);

train().then(() => {
    console.log('complete');
    let outputs = model.predict(xs);
    outputs.print();
})

async function train() {
    const config = {
        shuffle: true,
        epochs: 20
    }
    for (let i = 0; i < 200; i++) {
        const response = await model.fit(xs, ys, config);
        console.log(response.history.loss[0])
    }
}


