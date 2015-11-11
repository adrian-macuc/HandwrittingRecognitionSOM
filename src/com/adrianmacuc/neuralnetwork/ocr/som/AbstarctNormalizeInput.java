package com.adrianmacuc.neuralnetwork.ocr.som;

import com.adrianmacuc.neuralnetwork.ocr.matrix.Matrix;

public abstract class AbstarctNormalizeInput implements NormalizeInput{
	
	protected abstract Matrix createInputMatrix(final double[] pattern);
	
	protected abstract void calculateFactors(final double input[]);

}
