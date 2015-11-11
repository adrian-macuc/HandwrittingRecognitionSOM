package com.adrianmacuc.neuralnetwork.ocr.som;

import com.adrianmacuc.neuralnetwork.ocr.matrix.Matrix;
import com.adrianmacuc.neuralnetwork.ocr.matrix.MatrixAlgebra;
import com.adrianmacuc.neuralnetwork.ocr.matrix.MatrixImpl;
import com.adrianmacuc.neuralnetwork.ocr.som.NormalizeInput.NormalizationType;

public class SelfOrganizingMapImpl implements SelfOrganizingMap {

	private static final long serialVersionUID = 1L;

	protected Matrix outputWeights;
	protected double[] output;
	protected int inputNeuronCount;
	protected int outputNeuronCount;
	protected NormalizationType normalizationType;

	public SelfOrganizingMapImpl(final int inputCount, final int outputCount,
			final NormalizationType normalizationType) {
		this.inputNeuronCount = inputCount;
		this.outputNeuronCount = outputCount;
		this.outputWeights = new MatrixImpl(this.outputNeuronCount, this.inputNeuronCount + 1);
		this.output = new double[this.outputNeuronCount];
		this.normalizationType = normalizationType;
	}

	@Override
	public int getInputNeuronCount() {
		return this.inputNeuronCount;
	}

	@Override
	public double[] getOutputNeuronLayer() {
		return this.output;
	}

	@Override
	public int getOutputNeuronCount() {
		return this.outputNeuronCount;
	}

	@Override
	public Matrix getOutputNeuronLayerWeights() {
		return this.outputWeights;
	}

	@Override
	public void setOutputNeuronLayerWeights(Matrix outputWeights) {
		this.outputWeights = outputWeights;

	}

	@Override
	public NormalizationType getNormalizationType() {
		return this.normalizationType;
	}

	@Override
	public int winner(double[] input) {
		final NormalizeInput normalizedInput = new NormalizeInputImpl(input, this.normalizationType);
		return winner(normalizedInput);
	}

	@Override
	public int winner(NormalizeInput input) {
		int win = 0;

		double biggest = Double.MIN_VALUE;
		for (int i = 0; i < this.outputNeuronCount; i++) {
			final Matrix outputWeights = this.outputWeights.getRow(i);			
			this.output[i] = MatrixAlgebra.dotProduct(input.getInputMatrix(), outputWeights) * input.getNormalizationFactor();
			
			this.output[i] = (this.output[i] + 1.0) / 2.0;
			
			if (this.output[i] > biggest) {
				biggest = this.output[i];
				win = i;
			}

		
		}
		
		return win;
	}

}
