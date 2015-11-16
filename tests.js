

QUnit.test( "Neuron Tests", function( assert ) 
{
	// Neuron tests
	var neuron = new Neuron(3);
	neuron.weights = [1, 2, 3];
	neuron.bias = 2;
	console.log();
	console.log(neuron.calculateActivation([1, 1, 1]));

	assert.equal(neuron.calculateZ([1, 1, 1]), 8);
	var expected = 1 / (1 + math.exp(-8)); 
	assert.equal(neuron.calculateActivation([1, 1, 1]), expected);

	var prevDeltas = neuron.previousDeltas(2);
	assert.equal(prevDeltas.length, 3);
	assert.deepEqual(prevDeltas, [2, 4, 6]);

	neuron.applyDeltas([3, 2, 1], 4, 0.05);
	assert.deepEqual(neuron.weights, [0.3999999999999999, 1.6, 2.8]);
	
});

QUnit.test( "Layer Tests", function( assert ) 
{
	var layer = new Layer(2, 3);
	assert.equal(layer.neurons.length, 2);
	// assert.equal(layer.neurons[0].weights.length, 3);

	// layer = new Layer(2, 2);
	// layer.neurons[0].weights = [1, 3];
	// layer.neurons[1].weights = [2, 4];
	// layer.neurons[0].bias = 4;
	// layer.neurons[1].bias = 5;

	// assert.deepEqual(layer.getZArray([1, 0]), [1 - 4, 2 - 5]);
	// assert.deepEqual(layer.getZArray([0, 1]), [3 - 4, 4 - 5]);
	// assert.deepEqual(layer.getZArray([2, 4]), [10, 15]);

	// assert.deepEqual(layer.getActivationArray([1, 0]), [0.04742587317756678, 0.04742587317756678]);
	// assert.deepEqual(layer.getActivationArray([0, 1]), [0.2689414213699951, 0.2689414213699951]);
	// assert.deepEqual(layer.getActivationArray([2, 4]), [0.9999546021312976, 0.999999694097773]);
	
	// // Test previousDeltas()
	// assert.deepEqual(layer.neurons[0].previousDeltas(3), [3, 9]);
	// assert.deepEqual(layer.neurons[1].previousDeltas(5), [10, 20]);

	// assert.equal(sigmoidPrime(1), 0.19661193324148185);
	// assert.equal(sigmoidPrime(2), 0.10499358540350662);
	// assert.deepEqual(layer.previousDeltas([3, 5], [1, 2]), [2.555955132139264, 3.044813976701692]);

});


QUnit.test( "Network Tests", function( assert ) 
{
	// Following tutorial case: http://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
	var net = new Network([2, 2, 2]);
	net.layers[0].neurons[0].weights = [0.15, 0.20];
	net.layers[0].neurons[0].bias = 0.35;
	net.layers[0].neurons[1].weights = [0.25, 0.30];
	net.layers[0].neurons[1].bias = 0.35;

	net.layers[1].neurons[0].weights = [0.40, 0.45];
	net.layers[1].neurons[0].bias = 0.60;
	net.layers[1].neurons[1].weights = [0.50, 0.55];
	net.layers[1].neurons[1].bias = 0.60;

	assert.deepEqual(net.feedForward([0.05, 0.1]), [0.7513650695523157, 0.7729284653214625]);
	assert.deepEqual(sumCost(net.feedForward([0.05, 0.1]), [0.01, 0.99]), 0.2983711087600027);

	var finalZs = net.feedForwardZ([0.05, 0.1]);
	var finalAs = net.feedForward([0.05, 0.1]);
	console.log(finalAs);
	
	var cDerivative = [ costDerivative(finalAs[0], 0.01), costDerivative(finalAs[1], 0.99) ];
	var deltas = [];
	deltas[0] = [ cDerivative[0] * finalAs[0] * (1 - finalAs[0]), cDerivative[1] * finalAs[1] * (1 - finalAs[1]) ];
	deltas[1] = [ (0.40 * deltas[0][0] * 0.45) ];

	console.log(deltas);
	
	assert.deepEqual(cDerivative, [0.7413650695523157, -0.21707153467853746]);

	var z0 = [0.3775, 0.3925];
	console.log(deltas[0], z0);
	console.log(net.layers[1].neurons[0].weights);
	assert.deepEqual(net.layers[1].previousDeltas(deltas[0], z0), [0.008771354689486931, 0.009954254705217198]);
	console.log(net.layers[1].previousDeltas(deltas[0], z0));

	net.backprop([0.05, 0.1], [0.01, 0.99], 0.5);

	// Final layer weights
	assert.deepEqual(net.layers[1].neurons[0].weights, [0.35891647971788465, 0.4086661860762334]);
	assert.deepEqual(net.layers[1].neurons[1].weights, [0.5113012702387375, 0.5613701211079891]);

	//Hidden layer weights
	assert.deepEqual(net.layers[0].neurons[0].weights, [0.1497807161327628, 0.19956143226552567]);
	assert.deepEqual(net.layers[0].neurons[1].weights, [0.24975114363236958, 0.29950228726473915]);

	// assert.deepEqual(sumCost(net.feedForward([0.05, 0.1]), [0.01, 0.99]), 0.291027924);

	// for (var i = 0; i < 20000; i++)
	// {
	// 	net.backprop([0.05, 0.1], [0.01, 0.99], 0.5);
	// }

	// // assert.deepEqual(sumCost(net.feedForward([0.05, 0.1]), [0.01, 0.99]), 0.000035085);
	// console.log(net.feedForward([0.05, 0.1]));

});