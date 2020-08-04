package cotrainer;

import Jama.Matrix;
import java.io.FileReader;
import java.util.Arrays;
import java.util.PrimitiveIterator.OfDouble;

import common.*;
import regressor.KnnRegressor;
import weka.classifiers.pmml.consumer.Regression;
import weka.core.*;
import weka.gui.beans.TrainingSetEvent;

public class Cotrainer {
	/**
	 * The trainingSet.
	 */
	Instances trainingSet;
	/**
	 * The testingSet.
	 */
	Instances testingSet;
	/**
	 * The debugTrainingSet.
	 */
	Instances debugTrainingSet;
	/**
	 * The preDenosingSet.
	 */
	Instances preDenosingSet;
	/**
	 * The unlabeledSet.
	 */
	static Instances unlabeledSet;
	/**
	 * The knnRegressor.
	 */
	KnnRegressor knnRegressor;
	/**
	 * The kvalue of the cotrainer.
	 */
	int kValue;
	/**
	 * The poolSize.
	 */
	int poolSize;
	/**
	 * The threshold in select instances.
	 */
	double Threshold;
	/**
	 * The coefficient in linear regression.
	 */
	double[] X;
	/**
	 * The coefficient in linear regression.
	 */
	double y;
	/**
	 * The distanceMeasure of the Cotrainer.
	 */
	DistanceMeasure distanceMeasure;

	/**
	 ************************* 
	 * Get the trainingSetSize.
	 ************************* 
	 */
	public int trainingSetSize() {
		return trainingSet.numInstances();
	}

	public int debugtrainingSetSize() {
		return debugTrainingSet.numInstances();
	}

	public Instances getTrainingSet() {
		Instances resultInstances = new Instances(trainingSet);
		return resultInstances;
	}

	/**
	 ************************* 
	 * Get the kValue of the Cotrainer.
	 * 
	 * @return kValue
	 ************************* 
	 */
	public int getKvalue() {
		return kValue;
	}// of getKvalue

	/**
	 ************************* 
	 * Get the setkValue.
	 * 
	 * @param paraKvalue. The kValue of Cotrainer.
	 ************************* 
	 */
	public void setKvalue(int paraKvalue) {
		kValue = paraKvalue;
	}// of setKvalue

	/**
	 ************************* 
	 * Return the distanceMeasure of the Cotrainer.
	 * 
	 * @return distanceMeasure.
	 ************************* 
	 */
	public DistanceMeasure getDistanceMeasure() {
		return distanceMeasure;
	}// of getDistanceMeasure

	/**
	 ************************* 
	 * Set the distanceMeasure.
	 * 
	 * @param paraTrainingSet.     The trainingSet.
	 * @param paraDistanceMeasure. The number of distanceMeasure.
	 ************************* 
	 */
	public void setDistanceMeasure(Instances paraTrainingSet, int paraDistanceMeasure) {
		distanceMeasure = new DistanceMeasure(paraTrainingSet, paraDistanceMeasure);
	}// of setDistanceMeasure

	/**
	 ************************* 
	 * Get the poolSize of the unlabeledSet.
	 * 
	 * @return poolSize.
	 ************************* 
	 */
	public int getPoolSize() {
		return poolSize;
	}// ofgetPoolSize

	/**
	 ************************* 
	 * Set the poolSize of the unlabeledSet.
	 ************************* 
	 */
	public void setPoolSize(int paraPoolsize) {
		poolSize = paraPoolsize;
	}// of setPoolSize
	/**
	 ************************* 
	 * Set the self-pace-lamda
	 ************************* 
	 */
	public void setThreshold(double paraThreshold) {
		Threshold=paraThreshold;
	}// of setThreshold

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraTrainingSet     The given trainingSet.
	 * @param paraUnlabeledSet    The given unlabeledSet.
	 * @param paraTestingSet      The given testingSet.
	 * @param paraKvalue          The given kValue of the first cotraniner.
	 * @param paraPoolsize        The given poolsize of the unlableledSet.
	 * @param paraDistanceMeasure The distanceMeasure.
	 ********************
	 */
	public Cotrainer(int paraKvalue, Instances paraTrainingSet, Instances paraUnlabeledSet, Instances paraTestingSet,
			int paraPoolsize, int paraDistanceMeasure, double paraThreshold) {
		setKvalue(paraKvalue);
		setDistanceMeasure(paraTrainingSet, paraDistanceMeasure);
		setPoolSize(paraPoolsize);
		trainingSet = new Instances(paraTrainingSet);
		preDenosingSet = new Instances(paraTrainingSet);
		unlabeledSet = paraUnlabeledSet;
		testingSet = paraTestingSet;
		Threshold = paraThreshold;
		knnRegressor = new KnnRegressor(paraTrainingSet, testingSet, paraDistanceMeasure, paraKvalue);

	}// of cotrainer

	/**
	 ************************* 
	 * Select the most confidence instance from the poolSize of the unlabeledSet.
	 ************************* 
	 */
	public Instance selectCriticalInstance() {
		Instance tempInstance;
		Instance resultInstance;
		Instances tempTrainingSet;
		int[] tempPool=SimpleTools.getRandomSubset(unlabeledSet.numInstances(), unlabeledSet.numInstances());
		double[] tempDelta=new double[tempPool.length];
		double[] tempValue =new double[tempPool.length];
		
		/*if (poolSize >= unlabeledSet.numInstances()) {
			tempPool = new int[unlabeledSet.numInstances()];
			tempValue = new double[tempPool.length];
			for (int i = 0; i < unlabeledSet.numInstances(); i++) {
				tempPool[i] = i;
			}
			tempDelta = new double[tempPool.length];
		} else {
			tempPool = SimpleTools.getRandomSubset(unlabeledSet.numInstances(), unlabeledSet.numInstances());
			tempDelta = new double[tempPool.length];
			tempValue = new double[tempPool.length];
		}*/
		double maxDelta = -1;
		int tempIndex = -1;
		double tempClassValue = 0;
		double delta = 0;
		for (int i = 0; i < tempPool.length; i++) {
			tempInstance = new Instance(unlabeledSet.instance(tempPool[i]));
			tempTrainingSet = new Instances(trainingSet);
			// tempTrainingSet=trainingSet;
			tempClassValue = KnnRegressor.regression(kValue, trainingSet, tempInstance, distanceMeasure);
			tempInstance.setValue(tempInstance.numAttributes() - 1, tempClassValue);
			tempTrainingSet.add(tempInstance);
			KnnRegressor tempRegressor = new KnnRegressor(tempTrainingSet, testingSet, distanceMeasure, kValue);
			double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(tempInstance);
			int[] tempNeighbor = KnnRegressor.findNeighbor(kValue, trainingSet, tempUnlabelInstanceValue,
					distanceMeasure);
			double tempOldError = 0;
			double tempOldValue = 0;
			double tempNewError = 0;
			double tempNewValue = 0;
			for (int j = 0; j < tempNeighbor.length; j++) {
				tempOldError = knnRegressor.regression(trainingSet, trainingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempOldValue += tempOldError * tempOldError;
				tempNewError = tempRegressor.regression(tempTrainingSet, trainingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempNewValue += tempNewError * tempNewError;
			} // of for j
			delta = tempOldValue / tempNeighbor.length - tempNewValue / tempNeighbor.length;
			tempDelta[i] = delta;
			tempValue[i] = tempClassValue;
			if (delta > 0 && delta > maxDelta) {
				maxDelta = delta;
				tempIndex = i;
			} // of if
		} // of for i

		if (tempIndex == -1) {
			return null;
		} // of if
		resultInstance = new Instance(unlabeledSet.instance(tempPool[tempIndex]));
		tempClassValue = tempValue[tempIndex];
		resultInstance.setValue(resultInstance.numAttributes() - 1, tempClassValue);
		unlabeledSet.delete(tempPool[tempIndex]);
		return resultInstance;
	}// of selectCriticalInstances.£¨£©

	/**
	 ************************* 
	 * Select the most confidence instances from the poolSize of the unlabeledSet.
	 ************************* 
	 */
	public Instances selectCriticalInstances() {
		// step 1 Iinitialize and build the tool array
		Instance tempInstance;
		Instances resultInstances = new Instances(trainingSet, 0);
		Instances tempTrainingSet;
		int[] tempPool = SimpleTools.getRandomSubset(unlabeledSet.numInstances(), poolSize);
		double[] tempDelta = new double[tempPool.length];
		double[] tempValue = new double[tempPool.length];
		double maxDelta = -1;
		int tempIndex = -1;
		double tempClassValue = 0;
		double delta = 0;
		// step 2 Find the most confidence instance and calculate the change of the instance of the pool
		for (int i = 0; i < tempPool.length; i++) {
			tempInstance = new Instance(unlabeledSet.instance(tempPool[i]));
			tempTrainingSet = new Instances(trainingSet);
			// tempTrainingSet=trainingSet;
			tempClassValue = KnnRegressor.regression(kValue, trainingSet, tempInstance, distanceMeasure);
			tempInstance.setValue(tempInstance.numAttributes() - 1, tempClassValue);
			tempTrainingSet.add(tempInstance);
			KnnRegressor tempRegressor = new KnnRegressor(tempTrainingSet, testingSet, distanceMeasure, kValue);
			double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(tempInstance);
			int[] tempNeighbor = KnnRegressor.findNeighbor(kValue, trainingSet, tempUnlabelInstanceValue,
					distanceMeasure);
			double tempOldError = 0;
			double tempOldValue = 0;
			double tempNewError = 0;
			double tempNewValue = 0;
			for (int j = 0; j < tempNeighbor.length; j++) {
				tempOldError = knnRegressor.regression(trainingSet, trainingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempOldValue += tempOldError * tempOldError;
				tempNewError = tempRegressor.regression(tempTrainingSet, trainingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempNewValue += tempNewError * tempNewError;
			} // of for j
			delta = tempOldValue / tempNeighbor.length - tempNewValue / tempNeighbor.length;
			tempDelta[i] = delta;
			tempValue[i] = tempClassValue;
			if (delta > 0 && delta > maxDelta) {
				maxDelta = delta;
				tempIndex = i;
			} // of if
		} // of for i
		if (tempIndex == -1) {
			return null;
		} // of if
		// step 3 Find qualified instances in the result array and return them.
		int[] resultIndex = new int[tempDelta.length];
		Arrays.fill(resultIndex, 0);
		int k = 0;
		//step 3.1 find enough confidence instances in the pool.
		for (int i = 0; i < tempDelta.length; i++) {
			if (tempDelta[i] >= maxDelta * Threshold) {
				resultIndex[k] = tempPool[i];
				tempInstance = new Instance(unlabeledSet.instance(tempPool[i]));
				tempClassValue = tempValue[i];
				tempInstance.setValue(tempInstance.numAttributes() - 1, tempClassValue);
				resultInstances.add(tempInstance);
				k++;
			}
		}
		System.out.println(maxDelta + "\r\n");
		Arrays.sort(resultIndex);
		int[] reverse = new int[resultIndex.length];
		for (int i = 0; i < reverse.length; i++) {
			reverse[i] = resultIndex[resultIndex.length - i - 1];
		}
		for (int i = 0; i < reverse.length && reverse[i] != 0; i++) {
			unlabeledSet.delete(reverse[i]);
		}
		return resultInstances;
	}// of selectCriticalInstances.



	/**
	 ************************* 
	 * Update trainingSet.
	 ************************* 
	 */
	public void updateTrainingSet(Instance paraInstance) {
		trainingSet.add(paraInstance);
		knnRegressor.updatetrainingSet(trainingSet);
	}// of updateTrainingSet

	public void updateTrainingSet(Instances paraInstances) {
		for (int i = 0; i < paraInstances.numInstances(); i++) {
			updateTrainingSet(paraInstances.instance(i));
		}
	}

	/**
	 ************************* 
	 * Remove noise data from trainingSet.
	 ************************* 
	 */
	public void denoise() {
		// step.1 prepare the data for denoise.
		Instances tempTrainingSet = new Instances(trainingSet);
		int[] tempNeighbor = new int[kValue];
		Instance tempInstance;
		// tempTrainingSet = trainingSet;
		double[] tempMse = new double[(int) (tempTrainingSet.numInstances() * 0.05) + 2];
		int[] tempIndex = new int[(int) (tempTrainingSet.numInstances() * 0.05) + 2];
		int[] resultIndex = new int[(int) (tempTrainingSet.numInstances() * 0.05)];
		Arrays.fill(tempMse, 0);
		Arrays.fill(tempIndex, -1);
		tempMse[0] = Double.MAX_VALUE;
		// step.2 find the most nosing data from training set.
		for (int i = 0; i < tempTrainingSet.numInstances(); i++) {
			tempInstance = new Instance(tempTrainingSet.instance(i));
			tempNeighbor = knnRegressor.findNeighbor(kValue * 4, tempTrainingSet, i, tempInstance, distanceMeasure);
			double tempOldError = 0;
			double tempOldValue = 0;
			double tempNewError = 0;
			double tempNewValue = 0;
			for (int j = 0; j < tempNeighbor.length; j++) {
				tempOldError = knnRegressor.regression(trainingSet, trainingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempOldValue += tempOldError * tempOldError;
				tempNewError = knnRegressor.regression(tempTrainingSet, trainingSet.instance(tempNeighbor[j]), i,
						tempNeighbor[j]) - trainingSet.instance(tempNeighbor[j]).classValue();
				tempNewValue += tempNewError * tempNewError;
			} // of for j
			double tempMseerror = tempOldValue / tempNeighbor.length - tempNewValue / tempNeighbor.length;
			for (int k = (int) (tempTrainingSet.numInstances() * 0.05);; k--) {
				if (tempMseerror > tempMse[k]) {
					tempIndex[k + 1] = tempIndex[k];
					tempMse[k + 1] = tempMse[k];
				} else {
					tempIndex[k + 1] = i;
					tempMse[k + 1] = tempMseerror;
					break;
				} // of if
			} // of for k
		} // of for i
			// step.3 return the index of the noising data.
		for (int i = 0; i < resultIndex.length && tempIndex[i + 1] != 0; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		int[] reverseArrays = new int[resultIndex.length];
		Arrays.sort(resultIndex);
		for (int i = 0; i < resultIndex.length; i++) {
			reverseArrays[reverseArrays.length - 1 - i] = resultIndex[i];
		} // of for i
			// System.out.println(Arrays.toString(resultIndex));
			// System.out.println(Arrays.toString(reverseArrays));
		for (int i = 0; i < reverseArrays.length; i++) {
			tempTrainingSet.delete(reverseArrays[i]);

		} // of for i
			// System.out.println(trainingSet.numInstances());
			// System.out.println(tempTrainingSet.numInstances());
			// System.out.println(Arrays.toString(reverseArrays));
		debugTrainingSet = new Instances(tempTrainingSet);
		debugTrainingSet = tempTrainingSet;

		// System.out.println(debugTrainingSet.numInstances());
	}// of denoise

	/**
	 ************************* 
	 * Remove noise data before Co-training from trainingSet.
	 ************************* 
	 */
	public void preDenoise() {
		// step.1 prepare the data for denoise.
		Instances tempTrainingSet = new Instances(preDenosingSet);
		int[] tempNeighbor = new int[kValue];
		Instance tempInstance;
		// tempTrainingSet = trainingSet;
		double[] tempMse = new double[(int) (tempTrainingSet.numInstances() * 0.1) + 2];
		int[] tempIndex = new int[(int) (tempTrainingSet.numInstances() * 0.1) + 2];
		int[] resultIndex = new int[(int) (tempTrainingSet.numInstances() * 0.1)];
		Arrays.fill(tempMse, 0);
		Arrays.fill(tempIndex, -1);
		tempMse[0] = Double.MAX_VALUE;
		// step.2 find the most nosing data from training set.
		for (int i = 0; i < tempTrainingSet.numInstances(); i++) {
			tempInstance = new Instance(tempTrainingSet.instance(i));
			tempNeighbor = knnRegressor.findNeighbor(kValue * 4, tempTrainingSet, i, tempInstance, distanceMeasure);
			double tempOldError = 0;
			double tempOldValue = 0;
			double tempNewError = 0;
			double tempNewValue = 0;
			for (int j = 0; j < tempNeighbor.length; j++) {
				tempOldError = knnRegressor.regression(preDenosingSet, preDenosingSet.instance(tempNeighbor[j]),
						tempNeighbor[j]) - preDenosingSet.instance(tempNeighbor[j]).classValue();
				tempOldValue += tempOldError * tempOldError;
				tempNewError = knnRegressor.regression(tempTrainingSet, preDenosingSet.instance(tempNeighbor[j]), i,
						tempNeighbor[j]) - preDenosingSet.instance(tempNeighbor[j]).classValue();
				tempNewValue += tempNewError * tempNewError;
			} // of for j
			double tempMseerror = tempOldValue / tempNeighbor.length - tempNewValue / tempNeighbor.length;
			for (int k = (int) (tempTrainingSet.numInstances() * 0.1);; k--) {
				if (tempMseerror > tempMse[k]) {
					tempIndex[k + 1] = tempIndex[k];
					tempMse[k + 1] = tempMse[k];
				} else {
					tempIndex[k + 1] = i;
					tempMse[k + 1] = tempMseerror;
					break;
				} // of if
			} // of for k
		} // of for i
			// step.3 return the index of the noising data.
		for (int i = 0; i < resultIndex.length && tempIndex[i + 1] != 0; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		int[] reverseArrays = new int[resultIndex.length];
		Arrays.sort(resultIndex);
		for (int i = 0; i < resultIndex.length; i++) {
			reverseArrays[reverseArrays.length - 1 - i] = resultIndex[i];
		} // of for i
			// System.out.println(Arrays.toString(resultIndex));
			// System.out.println(Arrays.toString(reverseArrays));
		for (int i = 0; i < reverseArrays.length; i++) {
			tempTrainingSet.delete(reverseArrays[i]);

		} // of for i
			// System.out.println(trainingSet.numInstances());
			// System.out.println(tempTrainingSet.numInstances());
			// System.out.println(Arrays.toString(reverseArrays));
		preDenosingSet = tempTrainingSet;

		// System.out.println(debugTrainingSet.numInstances());
	}// of denoise

	/**
	 ************************* 
	 * Remove noise data before Co-training from trainingSet.
	 ************************* 
	 */
	public double traniningSetMse() {
		double resultMse = 0;
		int[] tempNeighbor = new int[kValue];
		Instance tempInstance;
		Instance regressInstance;
		int[] tempPool = SimpleTools.getRandomSubset(trainingSet.numInstances(),
				(int) (0.2 * trainingSet.numInstances()));
		for (int i = 0; i < tempPool.length; i++) {
			tempInstance = new Instance(trainingSet.instance(tempPool[i]));
			tempNeighbor = knnRegressor.deNoiseFindNeighbor(kValue, trainingSet, tempPool[i], tempPool[i], tempInstance,
					distanceMeasure);
			double tempValue = 0;
			double tempError = 0;
			for (int j = 0; j < tempNeighbor.length; j++) {
				tempValue = knnRegressor.regression(trainingSet, trainingSet.instance(tempNeighbor[j]), tempNeighbor[j])
						- trainingSet.instance(tempNeighbor[j]).classValue();
				tempError += tempValue * tempValue;
			}
			resultMse += tempError / kValue;
		}
		resultMse /= tempPool.length;
		System.out.println(resultMse);
		return resultMse;

	}

	public void crossValid() {
		int bestK = 0;
		int tempK;
		int[] tempNeighbor;
		double tempMse;
		double tempErr;
		double tempValue;
		double bestMse = Double.MAX_VALUE;
		Instance tempInstance;
		for (int i = 2 * kValue; i > 0; i--) {
			tempErr = 0;
			tempK = i;
			for (int j = 0; j < trainingSet.numInstances(); j++) {
				tempInstance = new Instance(trainingSet.instance(j));
				tempNeighbor = new int[i];
				tempNeighbor = knnRegressor.deNoiseFindNeighbor(tempK, trainingSet, j, j, tempInstance,
						distanceMeasure);
				tempValue = 0;
				for (int k = 0; k < tempNeighbor.length; k++) {
					tempValue += trainingSet.instance(tempNeighbor[k]).classValue();
				}
				tempValue /= tempNeighbor.length;
				// System.out.print(tempInstance.classValue());
				tempErr += (tempInstance.value(tempInstance.numAttributes() - 1) - tempValue)
						* (tempInstance.value(tempInstance.numAttributes() - 1) - tempValue);
			}
			if (bestMse > tempErr) {
				bestMse = tempErr;
				bestK = i;
			}
			System.out.print("The tempbest K is" + bestK + "\r\n");
		}
		System.out.print("The best K is" + bestK + "\r\n");
		kValue = bestK;
	}

	/**
	 ************************* 
	 * Predict the value of given instance.
	 * 
	 * @param paraInstance.The given instance.
	 * @return the prediction of the instance.
	 ************************* 
	 */
	public double predict(Instance paraInstance) {
		return knnRegressor.regression(kValue, trainingSet, paraInstance, distanceMeasure);
	}

	public double deNoisePredict(Instance paraInstance) {
		return knnRegressor.regression(kValue, debugTrainingSet, paraInstance, distanceMeasure);
	}

	public double preDeNoisePredict(Instance paraInstance) {
		return knnRegressor.regression(kValue, preDenosingSet, paraInstance, distanceMeasure);
	}

	public double predict(Instances paraTraningSet, Instance paraInstance) {
		return knnRegressor.regression(kValue, paraTraningSet, paraInstance, distanceMeasure);
	}
}
