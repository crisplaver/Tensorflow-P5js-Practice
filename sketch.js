let records = [
    [255, 0, 0, 'red'],
    [0, 255, 0, 'green'],
    [0, 0, 255, 'blue'],
    [255, 255, 0, 'yellow'],
    [0, 255, 255, 'acua'],
    [255, 0, 255, 'pink'],
];


let labelList = [
    'red',
    'green',
    'blue',
    'yellow',
    'acua',
    'pink',
]

let testColors = [];
let xs, ys, xp;
let model;

let resolution = 200

function setup() {
    createCanvas(800, 800);


    cols = width / resolution;
    rows = height / resolution;

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            let color = [random(255) / 255, random(255) / 255, random(255) / 255];
            testColors.push(color)
        }
    }

    lossP = createP('loss');
    epochP = createP('epochP');
    for (let v of records) {
        createP(v)
    }

    let colors = [];
    let labels = [];
    for (let record of records) {
        let color = [record[0] / 255, record[1] / 255, record[2] / 255];
        colors.push(color);
        let label = labelList.indexOf(record[3]);
        labels.push(label)
    }

    xs = tf.tensor2d(colors);
    let labelt = tf.tensor1d(labels, 'int32');
    ys = tf.oneHot(labelt, labelList.length);
    labelt.dispose()

    model = tf.sequential();

    let hidden = tf.layers.dense({
        units: 3,
        activation: 'sigmoid',
        inputDim: 3
    });

    let output = tf.layers.dense({
        units: labelList.length,
        activation: 'softmax',
    });

    model.add(hidden);
    model.add(output);

    const lr = 0.2;
    const optimizer = tf.train.sgd(lr);

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy'
    })

    train();
    setInterval(() => {
        tf.tidy(() => {
            predict()

        })
    }, 1000)
}

async function train() {
    const config = {
        shuffle: true,
        epochs: 10000,
        callbacks: {
            onTrainBegin: () => console.log('training start'),
            onTrainEnd: () => console.log('training complete'),
            // onBatchEnd: tf.nextFrame,
            onEpochEnd: (num, logs) => {
                lossP.html('loss: ' + logs.loss)
                epochP.html('epoch: ' + num)
            }

        }
    }
    return await model.fit(xs, ys, config)
}

function draw() {
    point(0, 0)
}

function predict() {
    let xs = tf.tensor2d(testColors)
    let results = model.predict(xs);
    let data = results.arraySync()

    if (testColors.length > 0) {
        let index = 0;
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                let color = testColors[index];
                fill(color[0] * 255, color[1] * 255, color[2] * 255);
                rect(resolution * j, resolution * i, resolution, resolution)

                fill(255);
                textSize(16);
                textAlign(CENTER, CENTER)

                let arr = data[index];
                let max = Math.max.apply(Math, arr)
                let indexOfMax = arr.indexOf(max)

                let label = labelList[indexOfMax];
                text(label, resolution * j, resolution * i, resolution, resolution);

                textSize(14);
                let label2 = `[${Math.floor(color[0] * 255)} ${Math.floor(color[1] * 255)} ${Math.floor(color[2] * 255)}]`
                text(label2, resolution * j, resolution * i, resolution, resolution + 30);
                text((max * 100).toFixed(1), resolution * j, resolution * i, resolution, resolution + 60)
                index++;
            }
        }
    }
}


