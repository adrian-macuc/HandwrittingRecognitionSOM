package com.adrianmacuc.neuralnetwork.ocr.matrix;

import java.io.Serializable;
import java.util.Arrays;

public class MatrixImpl implements Matrix, Serializable {

	private static final long serialVersionUID = 1L;

	double[][] matrix;

	public static Matrix arrayToColumnMatrix(final double[] array) {
		final double[][] columnMatrix = new double[array.length][1];
		for (int row = 0; row < columnMatrix.length; row++) {
			columnMatrix[row][0] = array[row];
		}
		return new MatrixImpl(columnMatrix);
	};

	public static Matrix arrayToRowMatrix(final double[] array) {
		final double[][] rowMatrix = new double[1][array.length];
		rowMatrix[0] = Arrays.copyOf(array, array.length);
		return new MatrixImpl(rowMatrix);
	};

	public MatrixImpl(final double[][] source2DArray) {
		this.matrix = new double[source2DArray.length][source2DArray[0].length];
		for (int row = 0; row < getRowsNumber(); row++) {
			for (int col = 0; col < getColumnsNumber(); col++) {
				this.setValue(row, col, source2DArray[row][col]);
			}
		}
	}

	public MatrixImpl(int rows, int cols) {
		this.matrix = new double[rows][cols];
	}

	@Override
	public void add(int row, int col, double value) {
		validate(row, col);
		final double newValue = getValue(row, col) + value;
		setValue(row, col, newValue);
	}

	@Override
	public double getValue(int row, int col) {
		validate(row, col);
		return this.matrix[row][col];
	}

	@Override
	public void setValue(int row, int col, double value) {
		validate(row, col);
		if (Double.isInfinite(value) || Double.isNaN(value)) {
			throw new MatrixError("Invalid value: " + value);
		}
		this.matrix[row][col] = value;
	}

	@Override
	public Matrix getColumn(int col) {
		if (col > getColumnsNumber()) {
			throw new MatrixError(
					"Column index " + col + " does not exist! The matrix only has " + getColumnsNumber() + " columns!");
		}

		final double[][] newMatrix = new double[getRowsNumber()][1];

		for (int row = 0; row < getRowsNumber(); row++) {
			newMatrix[row][0] = this.matrix[row][col];
		}

		return new MatrixImpl(newMatrix);
	}

	@Override
	public int getColumnsNumber() {
		return this.matrix[0].length;
	}

	@Override
	public Matrix getRow(int row) {
		if (row > getRowsNumber()) {
			throw new MatrixError(
					"Row index " + row + " does not exist! The matrix only has " + getRowsNumber() + " rows!");
		}

		final double[][] newMatrix = new double[1][getColumnsNumber()];

		for (int col = 0; col < getColumnsNumber(); col++) {
			newMatrix[0][col] = this.matrix[row][col];
		}

		return new MatrixImpl(newMatrix);
	}

	@Override
	public int getRowsNumber() {
		return this.matrix.length;
	}

	@Override
	public int getSize() {
		return getColumnsNumber() * getRowsNumber();
	}

	@Override
	public double getSum() {
		double result = 0;
		for (int row = 0; row < getRowsNumber(); row++) {
			for (int col = 0; col < getColumnsNumber(); col++) {
				result += this.matrix[row][col];
			}
		}
		return result;
	}

	@Override
	public boolean isVector() {
		if (getRowsNumber() == 1 || getColumnsNumber() == 1) {
			return true;
		}
		return false;
	}

	@Override
	public boolean isZeroMatrix() {
		for (int row = 0; row < getRowsNumber(); row++) {
			for (int col = 0; col < getColumnsNumber(); col++) {
				if (this.matrix[row][col] != 0) {
					return false;
				}
			}
		}
		return true;
	}

	@Override
	public void randomize(double min, double max) {
		for (int row = 0; row < getRowsNumber(); row++) {
			for (int col = 0; col < getColumnsNumber(); col++) {
				this.matrix[row][col] = (Math.random() * (max - min)) + min;
			}
		}

	}

	@Override
	public void clear() {
		for (int row = 0; row < getRowsNumber(); row++) {
			for (int col = 0; col < getColumnsNumber(); col++) {
				this.matrix[row][col] = 0;
			}
		}

	}

	@Override
	public void validate(int row, int col) throws MatrixError {
		if ((row >= getRowsNumber()) || (row < 0)) {
			throw new MatrixError("The row " + row + " is out of bounds:" + getRowsNumber());
		}

		if ((col >= getColumnsNumber()) || (col < 0)) {
			throw new MatrixError("The column " + col + " is out of bounds:" + getColumnsNumber());
		}
	}

	@Override
	public Double[] toPackedArray() {
		final Double result[] = new Double[getSize()];

		int index = 0;
		for (int row = 0; row < getRowsNumber(); row++) {
			for (int col = 0; col < getColumnsNumber(); col++) {
				result[index++] = getValue(row, col);
			}
		}

		return result;
	}
}
