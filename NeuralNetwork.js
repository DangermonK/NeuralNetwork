
class Neuron {

	constructor(inputs) {
		this.weights = [];
		this.bias = Math.random();
		for(let i = 0; i < inputs; i++) {
			this.weights.push(Math.random() * 2 - 1);
		}
		this.inputs = [];		
	}

	load(node) {
		this.weights = node.weights;
		this.bias = node.bias;
	}

	loadRandomized(nodes) {
		for(const i in this.weights) {
			const index = Math.floor(Math.random() * nodes.length);
			this.weights[i] = nodes[index].weights[i] + (Math.random() * 2 - 1) * 0.1;
		}
        const index = Math.floor(Math.random() * nodes.length);
		this.bias = nodes[index].bias + (Math.random() * 2 - 1) * 0.1;
    }

	activation(x) {
		return (1 / (1 + Math.pow(Math.E, -x))) * 2 - 1;
	}

	feed(input) {

		this.inputs = input;

		let output = 0;
		for(const x in this.weights) {
			output += this.weights[x] * this.inputs[x];
		}

		output += this.bias;
		output = this.activation(output);
		return output;

	}

	backpropagate(error, lr) {

		for(const x in this.weights) {
			this.weights[x] += error * this.inputs[x] * lr;
		}
		this.bias += error * lr;
		
		const errors = [];
		for(const weight of this.weights) {
			errors.push(error / weight);
		}

		return errors;

	}

}

class Layer {

	constructor(nodes, inputs) {
		this.inputs = inputs;
		this.nodes = [];
		for(let i = 0; i < nodes; i++) {
			this.nodes.push(new Neuron(inputs));
		}
	}

	load(layer) {
		this.nodes = [];
		for(const x in layer.nodes) {
			this.nodes.push(new Neuron(this.inputs));
			this.nodes[x].load(layer.nodes[x]);
		}
	}

	loadRandomized(layers) {
		this.nodes = [];
		for(const x in layers[0].nodes) {
			this.nodes.push(new Neuron(this.inputs));
			const neurons = [];
			for(const l of layers) {
				neurons.push(l.nodes[x]);
			}
			this.nodes[x].loadRandomized(neurons);
		}

	}

	feed(input) {
		const output = [];
		for(const node of this.nodes) {
			output.push(node.feed(input));
		}
		return output;
	}

	backpropagate(errors, lr) {
		const out_error = [];
		for(const x in this.nodes) {
			out_error.push(this.nodes[x].backpropagate(errors[x], lr));
		}

		const avg_error = Array(this.nodes[0].weights.length).fill(0);
		const length = this.nodes.length;
		for(const x in out_error) {
			for(const y in out_error[x]) {
				avg_error[y] += out_error[x][y];
			}
		}

		for(const i in avg_error) {
			avg_error[i] = avg_error[i] / length;
		}

		return avg_error;
	}

}

class NeuralNetwork {

	constructor(inputs) {
		this.layers = [];
		this.inputs = inputs;
	}

	load(network) {
		this.layers = [];
		for(const x in network.layers) {
			this.addLayer(network.layers[x].nodes.length);
			this.layers[x].load(network.layers[x]);
		}
	}

	loadRandomized(networks) {
		this.layers = [];
		for(const x in networks[0].layers) {
			this.addLayer(networks[0].layers[x].nodes.length);
			const l = [];
			for(const n of networks) {
				l.push(n.layers[x]);
			}
			this.layers[x].loadRandomized(l);
		}
	}

	addLayer(nodes) {
		let input_length = 0;
		if(this.layers.length === 0) {
			input_length = this.inputs;
		} else {
			input_length = this.layers[this.layers.length - 1].nodes.length;
		}

		this.layers.push(new Layer(nodes, input_length));
	}

	feed(input) {
		for(const layer of this.layers) {
			input = layer.feed(input);
		}
		return input;
	}

	train(input, target, lr) {
		const output = this.feed(input);
		let errors = [];
		for(const x in output) {
			errors.push(target[x] - output[x]);
		}	

		for(let i = this.layers.length - 1; i >= 0; i--) {
			errors = this.layers[i].backpropagate(errors, lr);
		}
	}

}
