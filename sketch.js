const resolution = 40;
let cols;
let rows;

const train_xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
const train_ys = tf.tensor1d([0, 1, 1, 0]);

let model;

const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'sigmoid'
})

const output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
})

const optimizer = tf.train.adam(0.2);



function train() {
    const config = {
        suffle: true,
        epochs: 20
    }
    model.fit(train_xs, train_ys, config).then((res) => {
        console.log(res.history.loss[0])
        train();
    });
}

function setup() {
    createCanvas(800, 800);
    cols = width / resolution;
    rows = height / resolution;

    let inputs = [];
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let x1 = i / rows;
            let x2 = j / cols;
            inputs.push([x1, x2])
        }
    }
    xs = tf.tensor2d(inputs)

    model = tf.sequential();
    model.add(hidden);
    model.add(output);
    model.compile({
        optimizer: optimizer,
        loss: tf.losses.meanSquaredError
    })

    train();
}

function draw() {
    background(255);
    
    let ys = model.predict(xs);
    let y_values = ys.dataSync();
    ys.dispose();

    let index = 0;
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let br = y_values[index] * 255;
            fill(br);
            rect(resolution * j, resolution * i, resolution, resolution)

            fill(255 - br);
            textSize(12);
            textAlign(CENTER, CENTER)
            text(nf(y_values[index], 1, 2), resolution * j, resolution * i, resolution, resolution);
            index++;
        }
    }
}