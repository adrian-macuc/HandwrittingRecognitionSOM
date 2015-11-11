package com.adrianmacuc.neuralnetwork.ocr.som;

import com.adrianmacuc.neuralnetwork.ocr.matrix.Matrix;

public interface NormalizeInput {
	
	public enum NormalizationType {
		Z_AXIS, MULTIPLICATIVE;
	}
	
	public Matrix getInputMatrix();
	
	public double getSynth();
	
	public double getNormalizationFactor();
	
}
