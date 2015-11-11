package com.adrianmacuc.neuralnetwork.ocr.matrix;

public interface Matrix {
	
	public void add(final int row, final int col, final double value);
	
	public double getValue(final int row, final int col);
	
	public void setValue(final int row, final int col, final double value);
	
	public Matrix getColumn(final int col);
	
	public int getColumnsNumber();
	
	public Matrix getRow(final int row);
	
	public int getRowsNumber();
	
	public int getSize();
	
	public double getSum();
	
	public boolean isVector();
	
	public boolean isZeroMatrix();
	
	public Double[] toPackedArray();
	
	public void randomize(final double min, final double max);
	
	public void clear();
	
	public void validate(final int row, final int col) throws MatrixError;

	
}
