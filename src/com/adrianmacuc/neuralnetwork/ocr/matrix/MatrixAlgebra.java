package com.adrianmacuc.neuralnetwork.ocr.matrix;

public class MatrixAlgebra {

	private MatrixAlgebra() {
	}

	public static Matrix add(final Matrix a, final Matrix b) {
		if (a.getRowsNumber() != b.getRowsNumber()) {
			throw new MatrixError("Matrices must have the same number of rows and columns.  Matrix A has "
					+ a.getRowsNumber() + " rows and matrix B has " + b.getRowsNumber() + " rows.");
		}

		if (a.getColumnsNumber() != b.getColumnsNumber()) {
			throw new MatrixError("Matrices must have the same number of rows and columns.  Matrix A has "
					+ a.getColumnsNumber() + " columns and matrix B has " + b.getColumnsNumber() + " columns.");
		}

		int rowsNumber = a.getRowsNumber();
		int colsNumber = a.getColumnsNumber();

		final double[][] result = new double[rowsNumber][colsNumber];

		for (int row = 0; row < rowsNumber; row++) {
			for (int col = 0; col < colsNumber; col++) {
				result[row][col] = a.getValue(row, col) + b.getValue(row, col);
			}
		}

		return new MatrixImpl(result);
	}

	public static Matrix subtract(final Matrix a, final Matrix b) {
		if (a.getRowsNumber() != b.getRowsNumber()) {
			throw new MatrixError("Matrices must have the same number of rows and columns.  Matrix A has "
					+ a.getRowsNumber() + " rows and matrix B has " + b.getRowsNumber() + " rows.");
		}

		if (a.getColumnsNumber() != b.getColumnsNumber()) {
			throw new MatrixError("Matrices must have the same number of rows and columns.  Matrix A has "
					+ a.getColumnsNumber() + " columns and matrix B has " + b.getColumnsNumber() + " columns.");
		}

		int rowsNumber = a.getRowsNumber();
		int colsNumber = a.getColumnsNumber();

		final double[][] result = new double[rowsNumber][colsNumber];

		for (int row = 0; row < rowsNumber; row++) {
			for (int col = 0; col < colsNumber; col++) {
				result[row][col] = a.getValue(row, col) - b.getValue(row, col);
			}
		}

		return new MatrixImpl(result);
	}

	public static void copy(final Matrix source, final Matrix target) {
		for (int row = 0; row < source.getRowsNumber(); row++) {
			for (int col = 0; col < source.getColumnsNumber(); col++) {
				target.setValue(row, col, source.getValue(row, col));
			}
		}

	}

	public static Matrix deleteColumn(final Matrix source, final int columnIndexToDelete) {
		if (columnIndexToDelete >= source.getColumnsNumber()) {
			throw new MatrixError("Column index " + columnIndexToDelete + " doesn't exist. Matrix only has "
					+ source.getColumnsNumber() + " columns.");
		}
		final double[][] resultedMatrix = new double[source.getRowsNumber()][source.getColumnsNumber() - 1];

		for (int row = 0; row < source.getRowsNumber(); row++) {
			int targetColumn = 0;

			for (int col = 0; col < source.getColumnsNumber(); col++) {
				if (col != columnIndexToDelete) {
					resultedMatrix[row][targetColumn] = source.getValue(row, col);
					targetColumn++;
				}

			}

		}
		return new MatrixImpl(resultedMatrix);
	}

	public static Matrix deleteRow(final Matrix source, final int rowIndexToDelete) {
		if (rowIndexToDelete >= source.getRowsNumber()) {
			throw new MatrixError("Row index " + rowIndexToDelete + " doesn't exist. Matrix only has "
					+ source.getRowsNumber() + " rows.");
		}
		final double[][] resultedMatrix = new double[source.getRowsNumber() - 1][source.getColumnsNumber()];

		int targetRow = 0;
		for (int row = 0; row < source.getRowsNumber(); row++) {
			if (row != rowIndexToDelete) {
				for (int col = 0; col < source.getColumnsNumber(); col++) {
					resultedMatrix[targetRow][col] = source.getValue(row, col);
				}
				targetRow++;
			}
		}
		return new MatrixImpl(resultedMatrix);
	}

	public static Matrix divide(final Matrix source, final double value) {
		final double[][] resultedMatrix = new double[source.getRowsNumber()][source.getColumnsNumber()];
		for (int row = 0; row < source.getRowsNumber(); row++) {
			for (int col = 0; col < source.getColumnsNumber(); col++) {
				resultedMatrix[row][col] = source.getValue(row, col) / value;
			}
		}
		return new MatrixImpl(resultedMatrix);
	}

	public static double dotProduct(final Matrix a, final Matrix b) {
		if (!a.isVector() || !b.isVector()) {
			throw new MatrixError("Both matrices must be vectors.");
		}

		final Double aArray[] = a.toPackedArray();
		final Double bArray[] = b.toPackedArray();

		if (aArray.length != bArray.length) {
			throw new MatrixError("Both matrices must be of the same length.");
		}

		double result = 0.0;
		final int length = aArray.length;

		for (int i = 0; i < length; i++) {
			result += aArray[i] * bArray[i];
		}
	
		return result;
	}

	public static Matrix getIdentityMatrix(final int size) {
		if (size < 1) {
			throw new MatrixError("Identity matrix must be at least of size 1.");
		}

		final Matrix result = new MatrixImpl(size, size);

		for (int i = 0; i < size; i++) {
			result.setValue(i, i, 1);
		}

		return result;
	}

	public static Matrix multiply(final Matrix source, final double value) {
		final double[][] result = new double[source.getRowsNumber()][source.getColumnsNumber()];
		for (int row = 0; row < source.getRowsNumber(); row++) {
			for (int col = 0; col < source.getColumnsNumber(); col++) {
				result[row][col] = source.getValue(row, col) * value;
			}
		}
		return new MatrixImpl(result);
	}

	public static Matrix multiply(final Matrix a, final Matrix b) {
		if (a.getColumnsNumber() != b.getRowsNumber()) {
			throw new MatrixError("Number of columns on the first matrix must match the number of rows on the second.");
		}

		final double[][] resultedMatrix = new double[a.getRowsNumber()][b.getColumnsNumber()];

		for (int row = 0; row < a.getRowsNumber(); row++) {
			for (int col = 0; col < b.getColumnsNumber(); col++) {
				double value = 0.0;

				for (int i = 0; i < a.getColumnsNumber(); i++) {

					value += a.getValue(row, i) * b.getValue(i, col);
				}
				resultedMatrix[row][col] = value;
			}
		}

		return new MatrixImpl(resultedMatrix);
	}

	public static Matrix transpose(final Matrix source) {
		final double inverseMatrix[][] = new double[source.getColumnsNumber()][source.getRowsNumber()];

		for (int row = 0; row < source.getRowsNumber(); row++) {
			for (int col = 0; col < source.getColumnsNumber(); col++) {
				inverseMatrix[col][row] = source.getValue(row, col);
			}
		}

		return new MatrixImpl(inverseMatrix);
	}

	public static double vectorSquaredLength(final Matrix source) {
		if (!source.isVector()) {
			throw new MatrixError("The matrix must be a vector.");
		}
		final Double[] vector = source.toPackedArray();
		double sum = 0.0;
		for (int i = 0; i < vector.length; i++) {
			sum += vector[i] * vector[i];
		}
		return Math.sqrt(sum);
	}

}
