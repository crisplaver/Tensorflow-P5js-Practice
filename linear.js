
let xs = [];
let ys = [];
let mt, bt

learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate)

function setup() {
    createCanvas(400, 400);
    mt = tf.variable(tf.scalar(random(1)));
    bt = tf.variable(tf.scalar(random(1)));
}

function loss() {
    pred = predict(xs);
    labels = tf.tensor1d(ys);
    return pred.sub(labels).square().mean()
}

function predict(x) {
    let xts = tf.tensor1d(x);
    let yts = mt.mul(xts).add(bt);
    return yts
}

function mousePressed() {
    x = map(mouseX, 0, width, 0, 1);
    y = map(mouseY, 0, height, 0, 1);
    xs.push(x);
    ys.push(y);
}


function draw() {
    background(0);
    if (xs.length > 0) {
        for (let i = 0; i < xs.length; i++) {
            x = map(xs[i], 0, 1, 0, width);
            y = map(ys[i], 0, 1, 0, height);
            stroke(255);
            strokeWeight(4);
            point(x, y)
        }

        optimizer.minimize(() => loss());


        const lineX = [0, 1];
        const yts = tf.tidy(() => predict(lineX))
        let lineY = yts.dataSync();
        yts.dispose();

        let x1 = map(lineX[0], 0, 1, 0, width);
        let x2 = map(lineX[1], 0, 1, 0, width);

        let y1 = map(lineY[0], 0, 1, 0, height);
        let y2 = map(lineY[1], 0, 1, 0, height);

        strokeWeight(2);
        line(x1, y1, x2, y2)


    }

    console.log(tf.memory().numTensors)

}