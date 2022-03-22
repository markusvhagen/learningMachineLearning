/* Loading the data */
async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    })).filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;
}

/* Plotting and running */
async function run() {
  const data = await getData();
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: "The data plotted"},
    {values},
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300
    }
  );

  const model = createModel();
  tfvis.show.modelSummary({name: "Model Summary"}, model);

  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  await trainModel(model, inputs, labels);
  console.log("Done training");

  testModel(model, data, tensorData);
}

/* Creating a sequential model */
function createModel() {
  const model = tf.sequential();
  // Adding a single input layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid'}));
  // Adding an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));
  return model;
}

/* Converting to tensor, shuffling and normalizing the data */
function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data); // Shuffling the data

    // Now we convert the data to tensor
    const inputs = data.map(d => d.horsepower);
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Normalizing the data
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin
    }
  });
}

/* Training the model */
async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"]
  });

  const batchSize = 32;
  const epochs = 200;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      {name: "Performance"},
      ["loss", "mean square error"],
      { height: 200, callbacks: ["onEpochEnd"]}
    )
  })
}

/* Testing the model! */
function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generating predictions uniformly in the range (0,1)
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs.reshape([100,1]))

    // Unnormalize the data to get original range, by doing inverted min-max
    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]}
  });

  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));

  // Compare original points and predicted points
  tfvis.render.scatterplot(
    {name: "Predicted vs original data"},
    {values: [originalPoints, predictedPoints], series: ["Original", "Predicted"]},
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300
    }
  );
}

document.addEventListener("DOMContentLoaded", run)
