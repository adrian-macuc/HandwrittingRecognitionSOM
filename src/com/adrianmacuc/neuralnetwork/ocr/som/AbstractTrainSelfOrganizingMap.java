package com.adrianmacuc.neuralnetwork.ocr.som;

import com.adrianmacuc.neuralnetwork.ocr.matrix.Matrix;

public abstract class AbstractTrainSelfOrganizingMap implements TrainSelfOrganizingMap{
	
	protected abstract void adjustWeights();
	
	protected abstract void forceWin();
	
	protected abstract void normalizeWeight(final Matrix matrix, final int row);
}
