class Matrix {

	constructor(rows, cols, fill = 0) {
		this.matrix = new Array(rows);
		this.rows = rows;
		this.cols = cols;
		
		for(let i = 0; i < rows; i++) {
			this.matrix[i] = new Array(cols).fill(fill);
		}

	}

	set(row, col, value) {
		this.matrix[row][col] = value;
	}

	get(row, col) {
		return this.matrix[row][col];
	}

	static map(m, f) {
		const matrix = new Matrix(m.rows, m.cols);
		for(let i = 0; i < m.rows; i++) {
			for(let j = 0; j < m.cols; j++) {
				matrix.set(i, j, f(m.get(i, j)));
			}
		}
		return matrix;
	}

	static add(m1, m2) {
		if(m1.rows !== m2.rows || m1.cols !== m2.cols) {
			throw "Matrix dimensions need to be equal!";
		} else {
			const matrix = new Matrix(m1.rows, m1.cols);
			for(let i = 0; i < m1.rows; i++) {
				for(let j = 0; j < m1.cols; j++) {
					matrix.set(i, j, (m1.get(i, j) + m2.get(i, j)));
				}
			}
			return matrix;
		}
	}

	static subtract(m1, m2) {
		if(m1.rows !== m2.rows || m1.cols !== m2.cols) {
			throw "Matrix dimensions need to be equal!";
		} else {
			const matrix = new Matrix(m1.rows, m1.cols);
			for(let i = 0; i < m1.rows; i++) {
				for(let j = 0; j < m1.cols; j++) {
					matrix.set(i, j, (m1.get(i, j) - m2.get(i, j)));
				}
			}
			return matrix;
		}
	}

	static transpose(m) {
		const matrix = new Matrix(m.cols, m.rows);
		for(let i = 0; i < m.rows; i++) {
			for(let j = 0; j < m.cols; j++) {
				matrix.set(j, i, m.get(i, j));
			}
		}		
		return matrix;
	}

	static multiply(m1, m2) {
		if(m1.cols !== m2.rows) {
			throw "Matrix dimensions need to be set correct!";
		} else {
			const matrix = new Matrix(m1.rows, m2.cols);
			for(let i = 0; i < matrix.rows; i++) {
				for(let j = 0; j < matrix.cols; j++) {
					for(let n = 0; n < m1.cols; n++) {
						matrix.matrix[i][j] += m1.get(i, n) * m2.get(n, j);
					}
				}
			}
			return matrix;
		}
	}
}

class Layer {

	constructor(weightMatrix, biasMatrix, range = 1, offset = 0) {
		this.weights = weightMatrix;
		this.biases = biasMatrix;
		this.range = range;
		this.offset = offset;
	}

	initializeRandom() {
		for(let row = 0; row < this.weights.rows; row++) {
			for(let col = 0; col < this.weights.cols; col++) {
				this.weights.set(row, col, Math.random() * this.range + this.offset);
			}
			this.biases.set(row, 0, Math.random() * this.range + this.offset);
		} 
	}

	randomize(factor) {
 		this.weights = Matrix.map(this.weights, (x) => {return x + (Math.random() * this.range + this.offset) * factor;});
 		this.biases = Matrix.map(this.biases, (x) => {return x + (Math.random() * this.range + this.offset) * factor;});
	}

	activation(x) {
		return (1 / (1 - Math.exp(-x))) * this.range + this.offset;
	}

	derivative(x) {
		return this.activation(x) * (1 - this.activation(x));
	}

	feed(input) {
		const net = Matrix.add(Matrix.multiply(this.weights, input), this.biases);
		const output = Matrix.map(net, (x) => {return this.activation(x);});
		return output;
	}

}

class NeuralNetwork{

	constructor(inputs, layerMatrix, min = 0, max = 1) {
		this.layers = [];
		this.range = max - min;
		this.offset = min;

		this.layers[0] = new Layer(new Matrix(layerMatrix[0], inputs), new Matrix(layerMatrix[0], 1), this.range, this.offset);
		for(let i = 1; i < layerMatrix.length; i++) {
			this.layers[i] = new Layer(new Matrix(layerMatrix[i], layerMatrix[i - 1]), new Matrix(layerMatrix[i], 1), this.range, this.offset);
		}	

		for(let i = 0; i < this.layers.length; i++) {
			this.layers[i].initializeRandom();
		}	
	}

	randomize(factor) {
		for(let i = 0; i < this.layers.length; i++) {
			this.layers[i].randomize(factor);
		}
	}

	feed(inputs) {
		for(let i = 0; i < this.layers.length; i++) {
			inputs = this.layers[i].feed(inputs);
		}
		return inputs;
	}

	train(inputs, targets, lr) {
		
		const output = this.feed(inputs);

		const error = Matrix.subtract(output, targets);

		const deriv = Matrix.multiply(output, Matrix.map(output, (x) => {return 1 - x;}));

		const d = Matrix.map(Matrix.multiply(deriv, error), (x) => {return x * -lr;});

		let dw = new Matrix(inputs.rows, 1);
		for(let i = 0; i < inputs.rows; i++) {
			dw.set(i, 0, Matrix.map(d, (x) => {return x * inputs.get(i, 0);}).get(0, 0));
		}

		this.layers[0].weights = Matrix.add(this.layers[0].weights, Matrix.transpose(dw));
		this.layers[0].biases = Matrix.add(this.layers[0].biases, );
	}

}


const nn = new NeuralNetwork(5, [1]);

const input = new Matrix(5, 1);
const target = new Matrix(1, 1);

input.set(0, 0, 1);
input.set(1, 0, 0);

target.set(0, 0, 1.8);

console.log(nn.feed(input));

for(let i = 0; i < 1000; i++) {
	nn.train(input, target, 0.1);
}

console.log(nn.feed(input));

