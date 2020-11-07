package cotrainer;

import java.util.Arrays;
import common.*;
import regressor.KnnRegressor;
import weka.core.*;

/**
 * The superclass of any co-trainer. It selects the most confidential instances for the partner. At
 * the same time, it accepts the suggestions of the partner.
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * @author Yu Li<br>
 *         Email:1132559357@qq.com<br>
 *         Date Created£ºAugust 5, 2020 <br>
 *         Last Modifide: August 8, 2020 <br>
 * 
 * @version 1.0
 */
public class Cotrainer {
	/**
	 * The training set.
	 */
	Instances trainingSet;

	/**
	 * The testing set.
	 */
	Instances testingSet;

	/**
	 * The unlabeled set.
	 */
	static Instances unlabeledSet;

	/**
	 * The training set that removed noisy data.
	 */
	Instances debugTrainingSet;

	/**
	 * The kNN regressor.
	 */
	KnnRegressor knnRegressor;

	/**
	 * The kValue of the cotrainer.
	 */
	int kValue;

	/**
	 * The pool size.
	 */
	int poolSize = 100;

	/**
	 * The lambda to select instances.
	 */
	double lambda;

	/**
	 * The distance measure (such as Manhattan, Euclidean distance) of the Cotrainer.
	 */
	DistanceMeasure distanceMeasure;

	/**
	 ************************* 
	 * Get the size of training set.
	 * 
	 * @return trainingSet.numInstances() The size of training set.
	 ************************* 
	 */
	public int getTrainingSetSize() {
		return trainingSet.numInstances();
	}// Of getTrainingSetSize

	/**
	 ************************* 
	 * Get the size from training set that removed data.
	 * 
	 * @return debugTrainingSet.numInstances() The size of training set that removed noisy data.
	 ************************* 
	 */
	public int getDebugTrainingSetSize() {
		return debugTrainingSet.numInstances();
	}// Of getDebugTrainingSetSize

	/**
	 ************************* 
	 * Get a copy of training set.
	 * 
	 * @return returnInstances A copy of training set.
	 ************************* 
	 */
	public Instances getTrainingSet() {
		Instances returnInstances = new Instances(trainingSet);
		return returnInstances;
	}// Of getTrainingSet

	/**
	 ************************* 
	 * Get the kValue of the cotrainer.
	 * 
	 * @return kValue
	 ************************* 
	 */
	public int getkValue() {
		return kValue;
	}// Of getkValue

	/**
	 ************************* 
	 * Set the kValue of cotrainer.
	 * 
	 * @param parakValue The kValue of cotrainer.
	 ************************* 
	 */
	public void setkValue(int parakValue) {
		kValue = parakValue;
	}// Of setkValue

	/**
	 ************************* 
	 * Return the distance measure of the Cotrainer.
	 * 
	 * @return distanceMeasure.
	 ************************* 
	 */
	public DistanceMeasure getDistanceMeasure() {
		return distanceMeasure;
	}// Of getDistanceMeasure

	/**
	 ************************* 
	 * Set the distanceMeasure.
	 * 
	 * @param paraTrainingSet     The trainingSet.
	 * @param paraDistanceMeasure The number of distanceMeasure.
	 ************************* 
	 */
	public void setDistanceMeasure(Instances paraTrainingSet, int paraDistanceMeasure) {
		distanceMeasure = new DistanceMeasure(paraTrainingSet, paraDistanceMeasure);
	}// Of setDistanceMeasure

	/**
	 ************************* 
	 * Get the pool size of the unlabeled set.
	 * 
	 * @return poolSize.
	 ************************* 
	 */
	public int getPoolSize() {
		return poolSize;
	}// Of getPoolSize

	/**
	 ************************* 
	 * Set the pool size of the unlabeled set.
	 * 
	 * @param paraPoolsize The pool size unlabeled set to select unlabeled instance
	 ************************* 
	 */
	public void setPoolSize(int paraPoolsize) {
		poolSize = paraPoolsize;
	}// Of setPoolSize

	/**
	 ************************* 
	 * Set the self-pace-lambda
	 * 
	 * @param paraLambda The self-pace lambda to control the process of select unlabeled instance.
	 ************************* 
	 */
	public void setLambda(double paraLambda) {
		lambda = paraLambda;
	}// Of setLambda

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraTrainingSet     The given training set.
	 * @param paraUnlabeledSet    The given unlabeled set.
	 * @param paraTestingSet      The given testing set.
	 * @param parakValue          The given kValue of the first cotraniner.
	 * @param paraPoolsize        The given pool size of the unlabeled set.
	 * @param paraDistanceMeasure The distance measure.
	 * @param paralambda          The self-pace lambda.
	 ********************
	 */
	public Cotrainer(int parakValue, Instances paraTrainingSet, Instances paraUnlabeledSet, Instances paraTestingSet,
			int paraPoolsize, int paraDistanceMeasure, double paralambda) {
		setkValue(parakValue);
		setDistanceMeasure(paraTrainingSet, paraDistanceMeasure);
		setPoolSize(paraPoolsize);
		trainingSet = new Instances(paraTrainingSet);
		unlabeledSet = paraUnlabeledSet;
		testingSet = paraTestingSet;
		lambda = paralambda;
		knnRegressor = new KnnRegressor(paraTrainingSet, testingSet, paraDistanceMeasure, parakValue);
	}// Of constructor Cotrainer

	/**
	 ************************* 
	 * Select the most confidence instances from the poolSize of the unlabeledSet.
	 * 
	 * @return resultInstace The most confidence unlabeled instance.
	 ************************* 
	 */
	public Instance selectCriticalInstance() {
		// Step 1 Initialize the parameter and build the tool array
		Instance tempInstance;
		Instance resultInstance;
		Instances tempTrainingSet;

		// Step 1.1 Build the pool of unlabeled set.
		int[] tempPool = SimpleTools.getRandomSubset(unlabeledSet.numInstances(), unlabeledSet.numInstances());
		double[] tempDelta = new double[tempPool.length];
		double[] tempValue = new double[tempPool.length];
		double maxDelta = -1;
		int tempIndex = -1;
		double tempClassValue = 0;
		double delta = 0;
		double tempOldError = 0;
		double tempOldValue = 0;
		double tempNewError = 0;
		double tempNewValue = 0;

		// Step 2 Compute the pseudo label and the reduction of mean square error from labeled neighborhood.
		for (int i = 0; i < tempPool.length; i++) {
			tempOldError = 0;
			tempOldError = 0;
			tempNewError = 0;
			tempNewValue = 0;
			tempInstance = new Instance(unlabeledSet.instance(tempPool[i]));
			tempTrainingSet = new Instances(trainingSet);

			// Step 2.1 Copy the unlabeled instance
			tempClassValue = KnnRegressor.regression(kValue, trainingSet, tempInstance, distanceMeasure);
			tempInstance.setValue(tempInstance.numAttributes() - 1, tempClassValue);
			tempTrainingSet.add(tempInstance);
			KnnRegressor tempRegressor = new KnnRegressor(tempTrainingSet, testingSet, distanceMeasure, kValue);
			double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(tempInstance);

			// Step2.2 Find the neighborhood of instance in labeled set and compute the mean squared error.
			int[] tempNeighbor = KnnRegressor.findNeighbor(kValue, trainingSet, tempUnlabelInstanceValue,
					distanceMeasure);
			for (int j = 0; j < tempNeighbor.length; j++) {
				tempOldError = knnRegressor.regression(trainingSet, trainingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempOldValue += tempOldError * tempOldError;
				tempNewError = tempRegressor.regression(tempTrainingSet, trainingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempNewValue += tempNewError * tempNewError;
			} // Of for j

			// Step 2.3 Compute the reduction of mean square error of labeled neighborhood.
			delta = tempOldValue / tempNeighbor.length - tempNewValue / tempNeighbor.length;
			tempDelta[i] = delta;
			tempValue[i] = tempClassValue;

			// Step 2.4 Select the confidence instance
			if (delta > 0 && delta > maxDelta) {
				maxDelta = delta;
				tempIndex = i;
			} // Of if
		} // Of for i

		// Step 3 Return the pseudo label
		if (tempIndex == -1) {
			return null;
		} // Of if
		resultInstance = new Instance(unlabeledSet.instance(tempPool[tempIndex]));
		tempClassValue = tempValue[tempIndex];
		resultInstance.setValue(resultInstance.numAttributes() - 1, tempClassValue);
		unlabeledSet.delete(tempPool[tempIndex]);
		return resultInstance;
	}// Of selectCriticalInstances.

	/**
	 ************************* 
	 * Select the most confidence instances from the poolSize of the unlabeledSet.
	 * 
	 * @return resultInstaces The most confidence unlabeled instances.
	 ************************* 
	 */
	public Instances selectCriticalInstances() {
		// Step 1 Initialize and build the tool array
		Instance tempInstance;
		Instances resultInstances = new Instances(trainingSet, 0);
		Instances tempTrainingSet;

		// Step 1.1 Build the pool of unlabeled set.
		int[] tempPool = SimpleTools.getRandomSubset(unlabeledSet.numInstances(), poolSize);
		double[] tempDelta = new double[tempPool.length];
		double[] tempValue = new double[tempPool.length];
		double maxDelta = -1;
		int tempIndex = -1;
		double tempClassValue = 0;
		double delta = 0;
		double tempOldError = 0;
		double tempOldValue = 0;
		double tempNewError = 0;
		double tempNewValue = 0;

		// Step 2 Compute the pseudo label and the reduction of mean square error from labeled neighborhood.
		for (int i = 0; i < tempPool.length; i++) {
			tempOldError = 0;
			tempOldValue = 0;
			tempNewError = 0;
			tempNewValue = 0;
			tempInstance = new Instance(unlabeledSet.instance(tempPool[i]));
			tempTrainingSet = new Instances(trainingSet);

			// Step 2.1 Copy the unlabeled instance
			tempClassValue = KnnRegressor.regression(kValue, trainingSet, tempInstance, distanceMeasure);
			tempInstance.setValue(tempInstance.numAttributes() - 1, tempClassValue);
			tempTrainingSet.add(tempInstance);
			KnnRegressor tempRegressor = new KnnRegressor(tempTrainingSet, testingSet, distanceMeasure, kValue);
			double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(tempInstance);

			// Step 2.2 Find the neighborhood of instance in labeled set and compute the mean squared error.
			int[] tempNeighbor = KnnRegressor.findNeighbor(kValue, trainingSet, tempUnlabelInstanceValue,
					distanceMeasure);
			for (int j = 0; j < tempNeighbor.length; j++) {
				tempOldError = knnRegressor.regression(trainingSet, trainingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempOldValue += tempOldError * tempOldError;
				tempNewError = tempRegressor.regression(tempTrainingSet, trainingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempNewValue += tempNewError * tempNewError;
			} // Of for j

			// Step 2.3 Compute the reduction of mean square error of labeled neighborhood.
			delta = tempOldValue / tempNeighbor.length - tempNewValue / tempNeighbor.length;
			tempDelta[i] = delta;
			tempValue[i] = tempClassValue;

			// Step 2.4 Select the confidence instance
			if (delta > 0 && delta > maxDelta) {
				maxDelta = delta;
				tempIndex = i;
			} // Of if
		} // Of for i
		if (tempIndex == -1) {
			return null;
		} // Of if
			// Step 3 Find qualified instances in the result array and return them.
		int[] resultIndex = new int[tempDelta.length];
		Arrays.fill(resultIndex, 0);
		int k = 0;
		// Step 3.1 Find enough confidence instances in the pool and store them.
		for (int i = 0; i < tempDelta.length; i++) {
			if (tempDelta[i] >= maxDelta * lambda) {
				resultIndex[k] = tempPool[i];
				tempInstance = new Instance(unlabeledSet.instance(tempPool[i]));
				tempClassValue = tempValue[i];
				tempInstance.setValue(tempInstance.numAttributes() - 1, tempClassValue);
				resultInstances.add(tempInstance);
				k++;
			} // Of if
		} // Of for i
			// Step 4 Return the pseudo label
		Arrays.sort(resultIndex);
		int[] deleteIndex = new int[resultIndex.length];
		for (int i = 0; i < deleteIndex.length; i++) {
			deleteIndex[i] = resultIndex[resultIndex.length - i - 1];
		}
		for (int i = 0; i < deleteIndex.length && deleteIndex[i] != 0; i++) {
			unlabeledSet.delete(deleteIndex[i]);
		}
		return resultInstances;
	}// Of selectCriticalInstances.

	/**
	 ************************* 
	 * Select the most confidence instances from the poolSize of the unlabeledSet.
	 * 
	 * @param paraCotraininerSet   The training set in another cotrainier.
	 * @param paraGamma            The adoptive value array based gamma.
	 * @param paraCortrainerkValue The kVlaue in another cotrainier.
	 * @return resultInstaces The most confidence unlabeled instances.
	 ************************* 
	 */
	public Instances selectCriticalInstances(Instances paraCotraininerSet, double[] paraGamma,
			int paraCortrainerkValue) {

		// Step 1 Initialize and build the tool array
		Instance tempInstance;
		Instances resultInstances = new Instances(trainingSet, 0);
		Instances tempFirstCortrainerTrainingSet;
		Instances tempSecondCortrainerTrainingSet;

		// Step 1.1 Build the pool of unlabeled set.
		int[] tempPool = SimpleTools.getRandomSubset(unlabeledSet.numInstances(), poolSize);
		// The confidence array of cotraininer.
		double[] tempFirstDelta = new double[tempPool.length];
		double[] tempSecondDelta = new double[tempPool.length];
		double[] tempValue = new double[tempPool.length];
		double MaxDelta = -1;
		int tempIndex = -1;
		double tempClassValue = 0;
		double firstDelta = 0;
		double secondDelta = 0;
		double tempFirstOldError = 0;
		double tempFirstOldValue = 0;
		double tempSecondOldError = 0;
		double tempSecondOldValue = 0;
		double tempFirstNewError = 0;
		double tempFirstNewValue = 0;
		double tempSecondNewError = 0;
		double tempSecondNewValue = 0;

		// Step 2 Compute the pseudo label and the reduction of mean square error from labeled neighborhood.
		for (int i = 0; i < tempPool.length; i++) {
			// Step 2.1 Initialize the tool parameters.
			tempFirstOldError = 0;
			tempFirstOldValue = 0;
			tempFirstNewError = 0;
			tempFirstNewValue = 0;
			tempSecondOldError = 0;
			tempSecondOldValue = 0;
			tempSecondNewError = 0;
			tempSecondNewValue = 0;
			tempInstance = new Instance(unlabeledSet.instance(tempPool[i]));
			tempFirstCortrainerTrainingSet = new Instances(trainingSet);
			tempSecondCortrainerTrainingSet = new Instances(paraCotraininerSet);

			// Step 2.2 Copy the unlabeled instance and added into different training sets.
			tempClassValue = KnnRegressor.regression(kValue, trainingSet, tempInstance, distanceMeasure);
			tempInstance.setValue(tempInstance.numAttributes() - 1, tempClassValue);
			tempFirstCortrainerTrainingSet.add(tempInstance);
			// Add instance into regressor.
			tempClassValue = KnnRegressor.regression(paraCortrainerkValue, paraCotraininerSet, tempInstance,
					distanceMeasure);
			tempInstance.setValue(tempInstance.numAttributes() - 1, tempClassValue);
			tempSecondCortrainerTrainingSet.add(tempInstance);

			// Step 2.3 Build the regressor using difference training sets.
			KnnRegressor tempFirstRegressor = new KnnRegressor(tempFirstCortrainerTrainingSet, testingSet,
					distanceMeasure, kValue);
			KnnRegressor tempSecondRegressor = new KnnRegressor(tempSecondCortrainerTrainingSet, testingSet,
					distanceMeasure, paraCortrainerkValue);
			double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(tempInstance);

			// Step 2.4 Find the neighborhood of instance in labeled set of first cotrainer and compute the
			// mean squared error.
			int[] tempFirstNeighbor = KnnRegressor.findNeighbor(kValue, trainingSet, tempUnlabelInstanceValue,
					distanceMeasure);
			for (int j = 0; j < tempFirstNeighbor.length; j++) {
				tempFirstOldError = knnRegressor.regression(trainingSet, trainingSet.instance(tempFirstNeighbor[j]),
						tempFirstNeighbor[j]) - trainingSet.instance(tempFirstNeighbor[j]).classValue();
				tempFirstOldValue += tempFirstOldError * tempFirstOldError;
				tempFirstNewError = tempFirstRegressor.regression(tempFirstCortrainerTrainingSet,
						trainingSet.instance(tempFirstNeighbor[j]), tempFirstNeighbor[j])
						- trainingSet.instance(tempFirstNeighbor[j]).classValue();
				tempFirstNewValue += tempFirstNewError * tempFirstNewError;
			} // Of for j

			// Step 2.5 Find the neighborhood of instance in labeled set of second cotrainer and compute the
			// mean squared error.
			int[] tempSecondNeighbor = KnnRegressor.findNeighbor(paraCortrainerkValue, paraCotraininerSet,
					tempUnlabelInstanceValue, distanceMeasure);
			for (int j = 0; j < tempSecondNeighbor.length; j++) {
				tempSecondOldError = knnRegressor.regression(paraCotraininerSet,
						paraCotraininerSet.instance(tempSecondNeighbor[j]), tempSecondNeighbor[j])
						- paraCotraininerSet.instance(tempSecondNeighbor[j]).classValue();
				tempSecondOldValue += tempSecondOldError * tempSecondOldError;
				tempSecondNewError = tempSecondRegressor.regression(tempSecondCortrainerTrainingSet,
						paraCotraininerSet.instance(tempSecondNeighbor[j]), tempSecondNeighbor[j])
						- paraCotraininerSet.instance(tempSecondNeighbor[j]).classValue();
				tempSecondNewValue += tempSecondNewError * tempSecondNewError;
			} // Of for j

			// Step 2.6 Compute the reduction of mean square error of labeled neighborhood.
			firstDelta = tempFirstOldValue / tempFirstNeighbor.length - tempFirstNewValue / tempFirstNeighbor.length;
			secondDelta = tempSecondOldValue / tempSecondNeighbor.length
					- tempSecondNewValue / tempSecondNeighbor.length;
			tempFirstDelta[i] = firstDelta;
			tempSecondDelta[i] = secondDelta;
			tempValue[i] = tempClassValue;

			// Step 2.7 Select the most confidence instance before revising.
			if (firstDelta > 0 && firstDelta > MaxDelta) {
				MaxDelta = firstDelta;
				tempIndex = i;
			} // Of if
				// tempFirstDelta[i] = tempFirstDelta[i] + tempSecondDelta[i];
		} // Of for i
		int[] argSortIndex = SimpleTool.argsort(tempSecondDelta);

		// Step 2.8 Using gamma to fix the confidence array.
		for (int i = 0; i < tempPool.length; i++) {
			if (tempSecondDelta[argSortIndex[i]] > 0) {
				tempFirstDelta[argSortIndex[i]] = tempFirstDelta[argSortIndex[i]]
						+ paraGamma[i] * tempSecondDelta[argSortIndex[i]];
			} else {
				tempFirstDelta[argSortIndex[i]] = tempFirstDelta[argSortIndex[i]]
						+ paraGamma[i] * Math.abs(tempSecondDelta[argSortIndex[i]]);
			} // Of if
		} // Of for i
		if (tempIndex == -1) {
			return null;
		} // Of if

		// Step 3 Find qualified instances in the result array and return them.
		int[] resultIndex = new int[tempFirstDelta.length];
		Arrays.fill(resultIndex, 0);
		int k = 0;

		// Step 3.1 Find enough confidence instances in the pool and store them.
		for (int i = 0; i < tempFirstDelta.length; i++) {
			if (tempFirstDelta[i] >= MaxDelta * lambda) {
				resultIndex[k] = tempPool[i];
				tempInstance = new Instance(unlabeledSet.instance(tempPool[i]));
				tempClassValue = tempValue[i];
				tempInstance.setValue(tempInstance.numAttributes() - 1, tempClassValue);
				resultInstances.add(tempInstance);
				k++;
			} // Of if
		} // Of for i

		// Step 4 Return the pseudo label
		Arrays.sort(resultIndex);
		int[] deleteIndex = new int[resultIndex.length];
		for (int i = 0; i < deleteIndex.length; i++) {
			deleteIndex[i] = resultIndex[resultIndex.length - i - 1];
		}
		for (int i = 0; i < deleteIndex.length && deleteIndex[i] != 0; i++) {
			unlabeledSet.delete(deleteIndex[i]);
		}
		return resultInstances;
	}// Of selectCriticalInstances.

	/**
	 ************************* 
	 * Update trainingSet.
	 * 
	 * @param paraInstance Add the selected instance into training set.
	 ************************* 
	 */
	public void updateTrainingSet(Instance paraInstance) {
		trainingSet.add(paraInstance);
		knnRegressor.updatetrainingSet(trainingSet);
	}// Of updateTrainingSet

	/**
	 ************************* 
	 * Update trainingSet.
	 * 
	 * @param paraInstances Add the selected instances into training set.
	 ************************* 
	 */
	public void updateTrainingSet(Instances paraInstances) {
		for (int i = 0; i < paraInstances.numInstances(); i++) {
			updateTrainingSet(paraInstances.instance(i));
		}
	}// Of updateTrainingSet

	/**
	 ************************* 
	 * Find best kValue of cotrainer before co-training process.
	 ************************* 
	 */
	public void crossValid() {

		// Step 1 Initialize the parameters
		int bestK = 0;
		int tempK;
		int[] tempNeighbor;
		double tempError;
		double tempValue;
		double bestMse = Double.MAX_VALUE;
		Instance tempInstance;

		// Step 2 Compute the reduction of neighborhood to find nosing data from training set.
		for (int i = 2 * kValue; i > 0; i--) {
			tempError = 0;
			tempK = i;

			// Step 2.1 Find neighbor
			for (int j = 0; j < trainingSet.numInstances(); j++) {
				tempInstance = new Instance(trainingSet.instance(j));
				tempNeighbor = new int[i];
				tempNeighbor = KnnRegressor.deNoiseFindNeighbor(tempK, trainingSet, j, j, tempInstance,
						distanceMeasure);
				tempValue = 0;

				// Step 2.2 Compute the mean square error of labeled instances
				for (int k = 0; k < tempNeighbor.length; k++) {
					tempValue += trainingSet.instance(tempNeighbor[k]).classValue();
				} // Of for k
				tempValue /= tempNeighbor.length;
				tempError += (tempInstance.value(tempInstance.numAttributes() - 1) - tempValue)
						* (tempInstance.value(tempInstance.numAttributes() - 1) - tempValue);
			} // Of for j

			// Step 3 log the best kValue of the cotrainier
			if (bestMse > tempError) {
				bestMse = tempError;
				bestK = i;
			} // Of if
			System.out.print("The tempbest K is" + bestK + "\r\n");
		} // Of for i

		// Step 4 set the best kValue.
		System.out.print("The best K is" + bestK + "\r\n");
		kValue = bestK;
	}// Of crossValid

	/**
	 ************************* 
	 * Predict the value of given instance.
	 * 
	 * @param paraInstance. The given instance.
	 * @return The prediction of the instance.
	 ************************* 
	 */
	public double predict(Instance paraInstance) {
		return KnnRegressor.regression(kValue, trainingSet, paraInstance, distanceMeasure);
	}// Of predict

	/**
	 ************************* 
	 * Predict the value of given instance in training set that removed noisy data.
	 * 
	 * @param paraInstance. The given instance.
	 * @return The prediction of the instance.
	 ************************* 
	 */
	public double deNoisePredict(Instance paraInstance) {
		return KnnRegressor.regression(kValue, debugTrainingSet, paraInstance, distanceMeasure);
	}// Of predict

	/**
	 ************************* 
	 * Predict the value of given instance.
	 * 
	 * @param paraInstance.   The given instance.
	 * @param paraTraningSet. The given training set.
	 * @return The prediction of the instance.
	 ************************* 
	 */
	public double predict(Instances paraTraningSet, Instance paraInstance) {
		return KnnRegressor.regression(kValue, paraTraningSet, paraInstance, distanceMeasure);
	}// Of predict

}// Of class Cotrainer
