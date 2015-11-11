package com.adrianmacuc.neuralnetwork.ocr.som;

import java.io.Serializable;

import com.adrianmacuc.neuralnetwork.ocr.matrix.Matrix;
import com.adrianmacuc.neuralnetwork.ocr.som.NormalizeInput.NormalizationType;

public interface SelfOrganizingMap extends Serializable{
	
	public static final double VERYSMALL = 1.E-30;
	
	public int getInputNeuronCount();
	
	public double[] getOutputNeuronLayer();
	
	public int getOutputNeuronCount();
	
	public Matrix getOutputNeuronLayerWeights();
	
	public void setOutputNeuronLayerWeights(final Matrix outputWeights);
	
	public NormalizationType getNormalizationType();
	
	public int winner(final double[] input);
	
	public int winner(final NormalizeInput input);
	
}
