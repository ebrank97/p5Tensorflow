/* TensorFlow 6.5, 6.6- Layers API */
//This is the model
const _model = tf.sequential();

//create hidden layer
//Dense is a fully connected neural network
const hidden = tf.layers.dense({
  units: 4, //number of nodes
  inputShape: [2], // shape of input
  activation: 'sigmoid'
});
//Add the layer
_model.add(hidden);

//	Create another layer
const output = tf.layers.dense({
  units: 1,
  // here the input shape is "inferred from the previous layer"
  activation: 'sigmoid'
});
//Add the layer
_model.add(output);

//An optimizer using gradient descent
const _optimizer = tf.train.sgd(0.1);

//Configuring is done so we can compile the model
_model.compile({
  optimizer: _optimizer,
  loss: tf.losses.meanSquaredError
});

const xs = tf.tensor2d([
  [0, 0],
  [0.5, 0.5],
  [1, 1]
]);

const ys = tf.tensor2d([
  [1],
  [0.5],
  [0]
]);
// const history = _model.fit(xs, ys).then(response => console.log(response.history.loss[0]));

train().then(() => {
  console.log('training complete');
  const outputs = _model.predict(xs);
  outputs.print();
});

async function train() {
  for (let i = 0; i < 10; i++) {
    const config = {
      shuffle: true
    }
    const response = await _model.fit(xs, ys, config);
    console.log(response.history.loss[0]);
  }
}








// const xs = tf.tensor2d([
//   [0.25, 0.92],
//	 [0.5, 0.7],
//	 [0.4, 0.3],
//	 [0.1, 0.7]
// ]);
// const ys = _model.predict(xs);
// ys.print();







function setup() {
  noCanvas();

  /* TensorFlow 6.2-Tensors

  	  const data = tf.tensor([0, 0, 127, 255, 100, 127.5, 24, 54], [2, 2, 2], 'int32');
  	  data.print();
  	  console.log(data.toString());

  	  const values = [];
  	  for (let i = 0; i < 30; i++) {
  	    values[i] = random(0, 100);
  	  }

  	  const shape = [2, 5, 3];

  	  const tense = tf.tensor3d(values, shape, 'int32');
  	  console.log(tense.toString());

  */

  /* TensorFlow 6.3-Variables & Operations

		  const values = [];
		  for (let i = 0; i < 30; i++) {
		    values[i] = random(0, 100);
		  }

		  const shape = [2, 5, 3];

		  const tense = tf.tensor3d(values, shape, 'int32');

		  const vtense = tf.variable(tense);

		  console.log(vtense);
		  tense.data().then(stuff => console.log(stuff));
		  tense.print();
		  console.log(tense.dataSync());

		  const values = [];
		  for (let i = 0; i < 15; i++) {
		    values[i] = random(0, 100);
		  }

		  const shape = [5, 3];
		  const shapeB = [3, 5];

		  const a = tf.tensor2d(values, shape, 'int32');
		  const b = tf.tensor2d(values, shape, 'int32');
		  const bb = b.transpose();
		  const c = a.add(b);
		  const c = a.matMul(bb);
		  a.print();
		  b.print();
		  c.print();

	*/

  /* TensorFlow 6.4- Memory Management

		  const values = [];

		  for (let i = 0; i < 15; i++) {
		    values[i] = random(0, 100);
		  }

		  const shape = [5, 3];

		  const a = tf.tensor2d(values, shape, 'int32');
		  const b = tf.tensor2d(values, shape, 'int32');
		  const b_t = b.transpose();
		  const c = a.matMul(b_t);

		  // c.print();
		  const test = tf.tensor2d(values, shape);
		  tf.tidy(() => {
		    const a = tf.tensor2d(values, shape, 'int32');
		    const b = tf.tensor2d(values, shape, 'int32');
		    const b_t = tf.keep(b.transpose());
		    const c = a.matMul(b_t);
		    //Do something Meaningful
		  });

		  test.dispose();

		  b_t.print();

		  console.log(tf.memory().numTensors);

	*/


}


function draw() {

  /* Testing Memory Management

		  // const values = [];
			//
		  // for (let i = 0; i < 15; i++) {
		  //   values[i] = random(0, 100);
		  // }
			//
		  // const shape = [5, 3];
			//
			//
			//
		  // tf.tidy(() => {
		  //   const a = tf.tensor2d(values, shape, 'int32');
		  //   const b = tf.tensor2d(values, shape, 'int32');
		  //   const b_t = b.transpose();
		  //   const c = a.matMul(b_t);
		  //   //Do something Meaningful
		  // });

		  //c.print();

		  // a.dispose();
		  // b.dispose();
		  // c.dispose();
		  // b_t.dispose();

		  // console.log(tf.memory().numTensors);

  */
}