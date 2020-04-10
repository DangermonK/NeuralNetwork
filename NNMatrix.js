class Matrix {

	constructor(rows, cols) {
		this.matrix = new Array(rows);
		this.rows = rows;
		this.cols = cols;
		
		for(let i = 0; i < rows; i++) {
			this.matrix[i] = new Array(cols).fill(0);
		}

	}

	set(row, col, value) {
		this.matrix[row][col] = value;
	}

	get(row, col) {
		return this.matrix[row][col];
	}

	addRow(value) {
		this.matrix[this.rows] = new Array(this.cols).fill(value);
		this.rows++;
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

class NeuralNetwork{

	constructor(inputs, layerMatrix, min = -1, max = 1) {
		this.layers = [];
		this.range = max - min;
		this.offset = min;

		this.layers[0] = new Matrix(layerMatrix[0], inputs + 1);
		for(let i = 1; i < layerMatrix.length; i++) {
			this.layers[i] = new Matrix(layerMatrix[i], layerMatrix[i - 1] + 1);
		}	

		for(let i = 0; i < this.layers.length; i++) {
			for(let row = 0; row < this.layers[i].rows; row++) {
				for(let col = 0; col < this.layers[i].cols; col++) {
					this.layers[i].set(row, col, Math.random() * this.range + this.offset);
				}
			}
		}	
	}

	randomize(factor) {
		for(let i = 0; i < this.layers.length; i++) {
			this.layers[i] = Matrix.map(this.layers[i], (x) => {return x + (Math.random() * this.range + this.offset) * factor;});
		}
	}

	activation(x) {
		return (1 / (1 - Math.exp(-x))) * this.range + this.offset;
	}

	feed(inputs) {
		for(let i = 0; i < this.layers.length; i++) {
			inputs.addRow(1);
			inputs = Matrix.map(Matrix.multiply(this.layers[0], inputs), (x) => {return this.activation(x);});
		}
		return inputs;
	}

}
