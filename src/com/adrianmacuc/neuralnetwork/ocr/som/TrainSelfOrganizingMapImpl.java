package com.adrianmacuc.neuralnetwork.ocr.som;

import com.adrianmacuc.neuralnetwork.ocr.matrix.Matrix;
import com.adrianmacuc.neuralnetwork.ocr.matrix.MatrixAlgebra;
import com.adrianmacuc.neuralnetwork.ocr.matrix.MatrixImpl;

public class TrainSelfOrganizingMapImpl extends AbstractTrainSelfOrganizingMap {

	private final SelfOrganizingMap som;

	protected LearningMethod learnMethod;
	protected double learnRate;
	protected double reduction = .99;
	protected double totalError;
	protected double globalError;

	int[] won;
	double[][] train;

	private final int inputNeuronCount;
	private final int outputNeuronCount;

	private final SelfOrganizingMap bestnet;

	private double bestError;
	private Matrix work;
	private Matrix correction;

	public TrainSelfOrganizingMapImpl(final SelfOrganizingMap som, final double[][] train, LearningMethod learnMethod,
			double learnRate) {
		this.som = som;
		this.train = train;
		this.totalError = 1.0;
		this.learnMethod = learnMethod;
		this.learnRate = learnRate;

		this.outputNeuronCount = som.getOutputNeuronCount();
		this.inputNeuronCount = som.getInputNeuronCount();

		for (int tset = 0; tset < train.length; tset++) {
			final Matrix dptr = MatrixImpl.arrayToColumnMatrix(train[tset]);
			if (MatrixAlgebra.vectorSquaredLength(dptr) < SelfOrganizingMap.VERYSMALL) {
				throw new RuntimeException("Multiplicative normalization has null training case");
			}
		}

		this.bestnet = new SelfOrganizingMapImpl(this.inputNeuronCount, this.outputNeuronCount,
				this.som.getNormalizationType());

		this.won = new int[this.outputNeuronCount];
		this.correction = new MatrixImpl(this.outputNeuronCount, this.inputNeuronCount + 1);
		if (this.learnMethod == LearningMethod.ADDITIVE) {
			this.work = new MatrixImpl(1, this.inputNeuronCount + 1);
		} else {
			this.work = null;
		}

		initialize();
		this.bestError = Double.MAX_VALUE;
	}

	@Override
	public void evaluateErrors() {
		this.correction.clear();

		for (int i = 0; i < this.won.length; i++) {
			this.won[i] = 0;
		}

		this.globalError = 0.0;

		for (int tset = 0; tset < this.train.length; tset++) {
			final NormalizeInput input = new NormalizeInputImpl(this.train[tset], this.som.getNormalizationType());
			final int best = this.som.winner(input);

			this.won[best]++;
			final Matrix wptr = this.som.getOutputNeuronLayerWeights().getRow(best);

			double length = 0.0;
			double diff;

			for (int i = 0; i < this.inputNeuronCount; i++) {
				diff = this.train[tset][i] * input.getNormalizationFactor() - wptr.getValue(0, i);
				length += diff * diff;
				if (this.learnMethod == LearningMethod.SUBTRACTIVE) {
					this.correction.add(best, i, diff);
				} else {
					this.work.setValue(0, i,
							this.learnRate * this.train[tset][i] * input.getNormalizationFactor() + wptr.getValue(0, i));
				}
			}
			diff = input.getSynth() - wptr.getValue(0, this.inputNeuronCount);
			length += diff * diff;
			if (this.learnMethod == LearningMethod.SUBTRACTIVE) {
				this.correction.add(best, this.inputNeuronCount, diff);
			} else {
				this.work.setValue(0, this.inputNeuronCount,
						this.learnRate * input.getSynth() + wptr.getValue(0, this.inputNeuronCount));
			}

			if (length > this.globalError) {
				this.globalError = length;
			}

			if (this.learnMethod == LearningMethod.ADDITIVE) {
				normalizeWeight(this.work, 0);
				for (int i = 0; i <= this.inputNeuronCount; i++) {
					this.correction.add(best, i, this.work.getValue(0, i) - wptr.getValue(0, i));
				}
			}

		}

		this.globalError = Math.sqrt(this.globalError);

	}

	@Override
	public double getBestError() {
		return this.bestError;
	}

	@Override
	public double getTotalError() {
		return this.totalError;
	}

	@Override
	public void initialize() {
		this.som.getOutputNeuronLayerWeights().randomize(-1, 1);

		for (int i = 0; i < this.outputNeuronCount; i++) {
			normalizeWeight(this.som.getOutputNeuronLayerWeights(), i);
		}
	}

	@Override
	public void iteration() {
		evaluateErrors();

		this.totalError = this.globalError;

		if (this.totalError < this.bestError) {
			this.bestError = this.totalError;
			copyWeights(this.som, this.bestnet);
		}

		int winners = 0;
		for (int i = 0; i < this.won.length; i++) {
			if (this.won[i] != 0) {
				winners++;
			}
		}

		if ((winners < this.outputNeuronCount) && (winners < this.train.length)) {
			forceWin();
			return;
		}

		adjustWeights();

		if (this.learnRate > 0.01) {
			this.learnRate *= this.reduction;
		}

	}

	@Override
	protected void adjustWeights() {
		for (int i = 0; i < this.outputNeuronCount; i++) {

			if (this.won[i] == 0) {
				continue;
			}

			double f = 1.0 / this.won[i];
			if (this.learnMethod == LearningMethod.SUBTRACTIVE) {
				f = f * this.learnRate;
			}

			for (int j = 0; j <= this.inputNeuronCount; j++) {
				final double correction = f * this.correction.getValue(i, j);
				this.som.getOutputNeuronLayerWeights().add(i, j, correction);
			}
		}

	}

	@Override
	protected void forceWin() {
		int best, which = 0;

		final Matrix outputWeights = this.som.getOutputNeuronLayerWeights();
		
		double dist = Double.MAX_VALUE;
		for (int tset = 0; tset < this.train.length; tset++) {
			best = this.som.winner(this.train[tset]);
			final double output[] = this.som.getOutputNeuronLayer();
			
			if (output[best] < dist) {
				dist = output[best];
				which = tset;
			}
		}

		final NormalizeInput input = new NormalizeInputImpl(this.train[which],
				this.som.getNormalizationType());
		best = this.som.winner(input);
		final double output[] = this.som.getOutputNeuronLayer();

		dist = Double.MIN_VALUE;
		int i = this.outputNeuronCount;
		while ((i--) > 0) {
			if (this.won[i] != 0) {
				continue;
			}
			if (output[i] > dist) {
				dist = output[i];
				which = i;
			}
		}

		for (int j = 0; j < input.getInputMatrix().getColumnsNumber(); j++) {
			outputWeights.setValue(which, j, input.getInputMatrix().getValue(0,j));
		}

		normalizeWeight(outputWeights, which);

	}

	@Override
	protected void normalizeWeight(Matrix matrix, int row) {
		double len = MatrixAlgebra.vectorSquaredLength(matrix.getRow(row));
		len = Math.max(len, SelfOrganizingMap.VERYSMALL);

		len = 1.0 / len;
		for (int i = 0; i < this.inputNeuronCount; i++) {
			matrix.setValue(row, i, matrix.getValue(row, i) * len);
		}
		matrix.setValue(row, this.inputNeuronCount, 0);

	}

	private void copyWeights(final SelfOrganizingMap source, final SelfOrganizingMap target) {
		MatrixAlgebra.copy(source.getOutputNeuronLayerWeights(), target.getOutputNeuronLayerWeights());
	}

}
