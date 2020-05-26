class Matrix {

	constructor(rows, cols, fill = 0) {
		this.matrix = new Array(rows);
		this.rows = rows;
		this.cols = cols;
		
		for(let i = 0; i < rows; i++) {
			this.matrix[i] = new Array(cols).fill(fill);
		}

	}

	randomize(factor) {
		for(let row = 0; row < this.rows; row++) {
			for(let col = 0; col < this.cols; col++) {
				this.set(row, col, this.get(row, col) + (Math.random() * 2 - 1) * factor);
			}	
		}	
	}

	fillRandom(range = 1, offset = 0) {
		for(let row = 0; row < this.rows; row++) {
			for(let col = 0; col < this.cols; col++) {
				this.set(row, col, Math.random() * range + offset);
			}	
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

	static AToM(array) {
		const matrix = new Matrix(array.length, 1);
		for(let i = 0; i < array.length; i++) {
			matrix.set(i, 0, array[i]);
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

	static multiplyEach(m1, m2) {
		if(m1.rows !== m2.rows || m1.cols !== m2.cols) {
			throw "Matrix dimensions need to be equal!";
		} else {
			const matrix = new Matrix(m1.rows, m1.cols);
			for(let i = 0; i < m1.rows; i++) {
				for(let j = 0; j < m1.cols; j++) {
					matrix.set(i, j, (m1.get(i, j) * m2.get(i, j)));
				}
			}
			return matrix;
		}
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

	static load(matrix) {
		const m = new Matrix(matrix.rows, matrix.cols);
		m.matrix = matrix.matrix;
		return m;
	}
}

class Layer {

	constructor(inputs, layerMatrix, min = 0, max = 1) {

		this.weights = new Matrix(layerMatrix[0], inputs);
		this.biases = new Matrix(layerMatrix[0], 1);

		this.layer = null;

		this.range = max - min;
		this.offset = min;
		if(layerMatrix.length > 1) {
			const nextInput = layerMatrix[0];
			layerMatrix.splice(0, 1);
			this.layer = new Layer(nextInput, layerMatrix, max, min);
		}

	}

	load(layer) {

		this.weights = Matrix.load(layer.weights);
		this.biases = Matrix.load(layer.biases);

		this.range = layer.range;
		this.offset = layer.offset;

		if(layer.layer !== null) {
			this.layer = new Layer(this.weights.rows, this.weights.cols);
			this.layer.load(layer.layer);
		}

	}

	randomizeWeightOffset(factor) {
		this.weights.randomize(factor);
		this.biases.randomize(factor);
		if(this.layer !== null) {
			this.layer.randomizeLayer(factor);
		}
	}

	initializeRandom() {

		this.weights.fillRandom(this.range, this.offset);
		this.biases.fillRandom(this.range, this.offset);

		if(this.layer !== null) {
			this.layer.initializeRandom();
		}

	}

	activation(x) {
		return (1 / (1 + Math.exp(-x))) * this.range + this.offset;
	}

	derivative(x) {
		return x * (1 - x);
	}

	feedForward(input) {

		const output = Matrix.map(Matrix.add(Matrix.multiply(this.weights, input), this.biases), x => {return this.activation(x);});

		if(this.layer !== null) {
			return this.layer.feedForward(output);
		} else {
			return output;
		}

	}

	train(input, target, lr = 0.1) {

		const output = Matrix.map(Matrix.add(Matrix.multiply(this.weights, input), this.biases), x => {return this.activation(x);});

		const derivative = Matrix.map(output, x => {return this.derivative(x);});

		if(this.layer !== null) {

			const nextLayer = this.layer.train(output, target, lr);

			let delta = new Matrix(this.weights.rows, 1);
			for(let i = 0; i < this.weights.rows; i++) {
				let dt = 0;
				for(let n = 0; n < nextLayer.delta.rows; n++) {
					dt += nextLayer.weights.get(n, i) * nextLayer.delta.get(n, 0);
				}
				delta.set(i, 0, dt);
			}

			delta = Matrix.multiplyEach(delta, derivative);

			const deltaWeight = Matrix.map(Matrix.multiply(delta, Matrix.transpose(input)), x => {return x * -lr;});

			this.weights = Matrix.add(this.weights, deltaWeight);
			this.biases = Matrix.add(this.biases, Matrix.map(delta, x => {return x * -lr;}));

			return {delta: delta, weights: this.weights};

		} else {

			const error = Matrix.subtract(target, output);

			const delta = Matrix.multiplyEach(derivative, error);

			const deltaWeight = Matrix.map(Matrix.multiply(delta, Matrix.transpose(input)), x => {return x * -lr;});

			this.weights = Matrix.add(this.weights, deltaWeight);
			this.biases = Matrix.add(this.biases, Matrix.map(delta, x => {return x * -lr;}));

			return {delta: delta, weights: this.weights};

		}

	}

}

