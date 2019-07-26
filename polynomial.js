
let xs = [];
let ys = [];
let at, bt, ct

learningRate = 0.1;
const optimizer = tf.train.adam(learningRate)

function setup() {
    createCanvas(400, 400);
    at = tf.variable(tf.scalar(random(-1,1)));
    bt = tf.variable(tf.scalar(random(-1,1)));
    ct = tf.variable(tf.scalar(random(-1,1)));
}

function loss() {
    pred = predict(xs);
    labels = tf.tensor1d(ys);
    return pred.sub(labels).square().mean()
}

function predict(x) {
    let xts = tf.tensor1d(x);
    // y = ax^2 + bx + c
    const yts = xts.square().mul(at).add(xts.mul(bt)).add(ct);
    return yts
}

function mousePressed() {
    x = map(mouseX, 0, width, -1, 1);
    y = map(mouseY, 0, height, 1, -1);
    xs.push(x);
    ys.push(y);
}


function draw() {
    background(0);
    if (xs.length > 0) {
        for (let i = 0; i < xs.length; i++) {
            x = map(xs[i], -1, 1, 0, width);
            y = map(ys[i], -1, 1, height, 0);
            stroke(255);
            strokeWeight(4);
            point(x, y)
        }

        optimizer.minimize(() => loss());


        const curveX = [];
        for (let x = -1; x < 1.01; x += 0.05) {
            curveX.push(x)
        }
        const yts = tf.tidy(() => predict(curveX))
        let curveY = yts.dataSync();
        yts.dispose();

        beginShape();
        noFill();
        stroke(255);
        strokeWeight(2);
        for(let i = 0; i < curveX.length; i++){
            let x = map(curveX[i], -1, 1, 0, width);
            let y = map(curveY[i], -1, 1, height, 0);
            vertex(x,y);
        }
        endShape();


    }

    console.log(tf.memory().numTensors)

}