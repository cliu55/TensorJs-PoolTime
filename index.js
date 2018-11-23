import * as tf from "@tensorflow/tfjs"
import "@tensorflow/tfjs-node"
import iris from "./iris.json"
import irisTesting from "./iris-testing.json"
import express from "express"
import bodyparser from "body-parser"

const app = express()
app.use(bodyparser.json())
const port = 80

// convert/setup our data
const trainingData = tf.tensor2d(iris.map(item => [
  item.day, item.time,
]))
const outputData = tf.tensor2d(iris.map(item => [
  item.availability === true ? 1 : 0,
  item.availability === false ? 1 : 0,
]))

// build neural network
const model = tf.sequential()

model.add(tf.layers.dense({
  inputShape: [2],
  activation: "sigmoid",
  units: 3,
}))
model.add(tf.layers.dense({
  inputShape: [3],
  activation: "sigmoid",
  units: 2,
}))
model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 2,
}))
model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.06),
})
// train/fit our network
model.fit(trainingData, outputData, {epochs: 100})

app.get('/', (req, res) => {
  res.send("HELLO WORLD")
})

app.post('/', (req, res) => {
  const testingData = tf.tensor2d([[req.body.day, req.body.time]])
  let probability = model.predict(testingData).toString();
  res.send(probability.substring(probability.indexOf("0."), probability.indexOf(",")));
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
