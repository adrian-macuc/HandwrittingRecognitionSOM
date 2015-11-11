package com.adrianmacuc.neuralnetwork.ocr.som;

import com.adrianmacuc.neuralnetwork.ocr.matrix.Matrix;
import com.adrianmacuc.neuralnetwork.ocr.matrix.MatrixAlgebra;
import com.adrianmacuc.neuralnetwork.ocr.matrix.MatrixImpl;

public class NormalizeInputImpl extends AbstarctNormalizeInput {

	private final NormalizationType type;
	protected double normalizationFactor;
	protected double syntheticInput;
	protected Matrix inputMatrix;

	public NormalizeInputImpl(final double[] input, final NormalizationType type) {
		this.type = type;
		calculateFactors(input);
		this.inputMatrix = createInputMatrix(input);
	}	

	@Override
	protected Matrix createInputMatrix(double[] pattern) {
		final Matrix result = new MatrixImpl(1, pattern.length + 1);
		
		for (int i = 0; i < pattern.length; i++) {
			result.setValue(0, i, pattern[i]);
		}

		result.setValue(0, pattern.length, this.syntheticInput);

		return result;
	}

	@Override
	protected void calculateFactors(double[] input) {
		final Matrix inputMatrix = MatrixImpl.arrayToColumnMatrix(input);
		double len = MatrixAlgebra.vectorSquaredLength(inputMatrix);

		len = Math.max(len, SelfOrganizingMap.VERYSMALL);
		final int numInputs = input.length;

		if (this.type == NormalizationType.MULTIPLICATIVE) {
			this.normalizationFactor = 1.0 / len;
			this.syntheticInput = 0.0;
		} else {
			this.normalizationFactor = 1.0 / Math.sqrt(numInputs);
			final double d = numInputs - Math.pow(len, 2);
			if (d > 0.0) {
				this.syntheticInput = Math.sqrt(d) * this.normalizationFactor;
			} else {
				this.syntheticInput = 0;
			}
		}
	}

	@Override
	public Matrix getInputMatrix() {
		return this.inputMatrix;
	}
	
	@Override
	public double getNormalizationFactor() {
		return this.normalizationFactor;
	}

	@Override
	public double getSynth() {
		return this.syntheticInput;
	}
}
