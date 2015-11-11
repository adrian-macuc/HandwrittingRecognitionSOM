package com.adrianmacuc.neuralnetwork.ocr.som;

public interface TrainSelfOrganizingMap {

	public enum LearningMethod {
		ADDITIVE, SUBTRACTIVE;
	}
	
	public void evaluateErrors();
	
	public double getBestError();
	
	public double getTotalError();
	
	public void initialize();
	
	public void iteration();
}
