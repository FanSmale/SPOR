package regressor;

import java.io.FileReader;
import java.util.Arrays;
import common.*;
import weka.core.*;
/**
 * The kNN regressor used by SPOR. 
 * Project: The self-pace co-training regression.
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * @author Yu Li<br>
 *         Email:1132559357@qq.com<br>
 *         Date Created£ºAugust 5, 2020 <br>
 *         Last Modifide: August 8, 2020 <br>
 * 
 * @version 1.1
 */
public class KnnRegressor {
	/**
	 * The training set.
	 */
	Instances trainingSet;

	/**
	 * The testing set.
	 */
	Instances testingSet;

	/**
	 * The k value for kNN.
	 */
	int kValue;

	/**
	 * The distance measure for regressor.
	 */
	DistanceMeasure distanceMeasure;

	/**
	 * Get the distanceMeasure value for regressor.
	 * 
	 * @return DistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *         regressorr
	 */
	public DistanceMeasure getDistanceMeasure() {
		return distanceMeasure;
	}// of getDistanceMeasure

	/**
	 * Set the distanceMeasure value for regressor.
	 * 
	 * @param paraTrainingSet     The training set.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor
	 */
	public void setDistanceMeasure(Instances paraTrainingSet, int paraDistanceMeasure) {
		distanceMeasure = new DistanceMeasure(paraTrainingSet, paraDistanceMeasure);
	}// of setDistanceMeasure

	/**
	 * Get the k value for kNNregressor.
	 * 
	 * @return kValue The kValue for regressor.
	 */
	public int getkValue() {
		return kValue;
	}// of getkValue

	/**
	 * Set the k value for kNNregressor.
	 * 
	 * @param parakValue The k value for kNNregressor.
	 */
	public void setKvalue(int parakValue) {
		kValue = parakValue;
	}// of setkValue

	/**
	 * Update the traningSet.
	 * 
	 * @param paraInstances The newly traningSet.
	 */
	public void updatetrainingSet(Instances paraInstances) {
		trainingSet = paraInstances;
	}

	/**
	 * Build the regressor.
	 * 
	 * @param paraTraningSet The training set.
	 * @param paraTestingSet The testing set.
	 */
	public KnnRegressor(Instances paraTraningSet, Instances paraTestingSet) {
		trainingSet = paraTraningSet;
		testingSet = paraTestingSet;
	}// of constructor

	/**
	 * Build the regressor.
	 * 
	 * @param paraTraningSet      The training set.
	 * @param paraTestingSet      The testing set.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor
	 * @param parakValue          The kValue of KnnRegressor.
	 */
	public KnnRegressor(Instances paraTraningSet, Instances paraTestingSet, int paraDistanceMeasure, int parakValue) {
		this(paraTraningSet, paraTestingSet);
		setDistanceMeasure(trainingSet, paraDistanceMeasure);
		setKvalue(parakValue);
	}// of constructor

	/**
	 * Build the regressor.
	 * 
	 * @param paraTraningSet      The training set.
	 * @param paraTestingSet      The testing set.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            Cotrainer
	 * @param parakValue          The kValue of KnnRegressor.
	 */
	public KnnRegressor(Instances paraTraningSet, Instances paraTestingSet, DistanceMeasure paraDistanceMeasure,
			int parakValue) {
		this(paraTraningSet, paraTestingSet);
		distanceMeasure = paraDistanceMeasure;
		setKvalue(parakValue);
	}// of constructor

	/**
	 ********************
	 * Find the neighbor of the given instance.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraTraningSet      The training set of the regressor.
	 * @param paraInstance        The instance needs to find neighbor.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @return resultIndex The index of nearest neighborhood.
	 ********************
	 */
	public static int[] findNeighbor(int parakValue, Instances paraTraningSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure) {
		// Step 1. Initialize the parameters and tool array.
		double[] tempUnlabeleddata = DistanceMeasure.instanceToDoubleArray(paraInstance);
		int[] tempIndex = new int[parakValue + 2];
		int[] resultIndex = new int[parakValue];
		double[] tempDistance = new double[parakValue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;

		// Step 2. Compute the distance of training set.
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(tempUnlabeleddata, tempTraningValue);
			for (int j = parakValue;; j--) {
				if (tempCurrentDistance < tempDistance[j]) {
					tempIndex[j + 1] = tempIndex[j];
					tempDistance[j + 1] = tempDistance[j];
				} else {
					tempIndex[j + 1] = i;
					tempDistance[j + 1] = tempCurrentDistance;
					break;
				} // of if
			} // of for j
		} // of for i
			// Step 3. Sort the distance and return the index of neighborhood.
		for (int i = 0; i < parakValue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	/**
	 ********************
	 * Find the neighbor of the given instance.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraTraningSet      The training set of the regressor.
	 * @param paraInstance        The instance needs to find neighbor.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @param paraIndex           The index of given instance.
	 * @return resultIndex The index of nearest neighborhood.
	 ********************
	 */
	public static int[] findNeighbor(int parakValue, Instances paraTraningSet, int paraIndex, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure) {

		// Step 1. Initialize the parameters and tool array.
		double[] tempUnlabeleddata = DistanceMeasure.instanceToDoubleArray(paraInstance);
		int[] tempIndex = new int[parakValue + 2];
		int[] resultIndex = new int[parakValue];
		double[] tempDistance = new double[parakValue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;

		// Step 2. Compute the distance of training set.
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			if (i == paraIndex) {
				continue;
			} // of if
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(tempUnlabeleddata, tempTraningValue);
			for (int j = parakValue;; j--) {
				if (tempCurrentDistance < tempDistance[j]) {
					tempIndex[j + 1] = tempIndex[j];
					tempDistance[j + 1] = tempDistance[j];
				} else {
					tempIndex[j + 1] = i;
					tempDistance[j + 1] = tempCurrentDistance;
					break;
				} // of if
			} // of for j
		} // of for i

		// Step 3. Sort the distance and return the index of neighborhood.
		for (int i = 0; i < parakValue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	/**
	 ********************
	 * Find the neighbor of the given instance.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraTraningSet      The training set of the regressor.
	 * @param paraUnlabeleddata   The instance needs to find neighbor.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @return resultIndex The index of nearest neighborhood.
	 ********************
	 */
	public static int[] findNeighbor(int parakValue, Instances paraTraningSet, double[] paraUnlabeleddata,
			DistanceMeasure paraDistanceMeasure) {

		// Step 1. Initialize the parameters and tool array.
		int[] tempIndex = new int[parakValue + 2];
		int[] resultIndex = new int[parakValue];
		double[] tempDistance = new double[parakValue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;

		// Step 2. Compute the distance of training set.
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(paraUnlabeleddata, tempTraningValue);
			for (int j = parakValue;; j--) {
				if (tempCurrentDistance < tempDistance[j]) {
					tempIndex[j + 1] = tempIndex[j];
					tempDistance[j + 1] = tempDistance[j];
				} else {
					tempIndex[j + 1] = i;
					tempDistance[j + 1] = tempCurrentDistance;
					break;
				} // of if
			} // of for j
		} // of for i

		// Step 3. Sort the distance and return the index of neighborhood.
		for (int i = 0; i < parakValue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	/**
	 ********************
	 * Find the neighbor of the given instance.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraTraningSet      The training set of the regressor.
	 * @param paraUnlabeleddata   The instance needs to find neighbor.
	 * @param paraIndex           The index of given instance.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @return resultIndex The index of nearest neighborhood.
	 ********************
	 */
	public static int[] findNeighbor(int parakValue, Instances paraTraningSet, int paraIndex,
			double[] paraUnlabeleddata, DistanceMeasure paraDistanceMeasure) {

		// Step 1. Initialize the parameters and tool array.
		int[] tempIndex = new int[parakValue + 2];
		int[] resultIndex = new int[parakValue];
		double[] tempDistance = new double[parakValue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;

		// Step 2. Compute the distance of training set.
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			if (i == paraIndex) {
				continue;
			} // of if
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(paraUnlabeleddata, tempTraningValue);
			for (int j = parakValue;; j--) {
				if (tempCurrentDistance < tempDistance[j]) {
					tempIndex[j + 1] = tempIndex[j];
					tempDistance[j + 1] = tempDistance[j];
				} else {
					tempIndex[j + 1] = i;
					tempDistance[j + 1] = tempCurrentDistance;
					break;
				} // of if
			} // of for j
		} // of for i

		// Step 3. Sort the distance and return the index of neighborhood.
		for (int i = 0; i < parakValue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	/**
	 ********************
	 * Find the neighbor of the given instance.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraTraningSet      The training set of the regressor.
	 * @param paraInstance        The instance needs to find neighbor.
	 * @param paraIndex1          The index of given instance.
	 * @param paraIndex2          The index of training set instance.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @return resultIndex The index of nearest neighborhood.
	 ********************
	 */
	public static int[] deNoiseFindNeighbor(int parakValue, Instances paraTraningSet, int paraIndex1, int paraIndex2,
			Instance paraInstance, DistanceMeasure paraDistanceMeasure) {

		// Step 1. Initialize the parameters and tool array.
		double[] tempUnlabeleddata = DistanceMeasure.instanceToDoubleArray(paraInstance);
		int[] tempIndex = new int[parakValue + 2];
		int[] resultIndex = new int[parakValue];
		double[] tempDistance = new double[parakValue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;

		// Step 2. Compute the distance of training set.
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			if (i == paraIndex1 || i == paraIndex2) {
				continue;
			}
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(tempUnlabeleddata, tempTraningValue);
			for (int j = parakValue;; j--) {
				if (tempCurrentDistance < tempDistance[j]) {
					tempIndex[j + 1] = tempIndex[j];
					tempDistance[j + 1] = tempDistance[j];
				} else {
					tempIndex[j + 1] = i;
					tempDistance[j + 1] = tempCurrentDistance;
					break;
				} // of if
			} // of for j
		} // of for i

		// Step 3. Sort the distance and return the index of neighborhood.
		for (int i = 0; i < parakValue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	/**
	 ********************
	 * Predict the value of the given instance.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraTraningSet      The training set of the regressor.
	 * @param paraInstance        The instance needs to find neighbor.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @return resultPrediction The prediction of given instance.
	 ********************
	 */
	public static double knn(int parakValue, Instances paraTraningSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure) {
		int[] tempIndex = new int[parakValue + 2];
		double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(paraInstance);
		tempIndex = findNeighbor(parakValue, paraTraningSet, tempUnlabelInstanceValue, paraDistanceMeasure);
		double tempPrediction = 0;
		for (int i = 0; i < parakValue; i++) {
			tempPrediction += paraTraningSet.instance(tempIndex[i]).classValue();
		} // of for i
		tempPrediction = tempPrediction / parakValue;
		double resultPrediction = tempPrediction;
		return resultPrediction;
	}// of Knn

	/**
	 ********************
	 * Predict the value of the given instance.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraTraningSet      The training set of the regressor.
	 * @param paraInstance        The instance needs to find neighbor.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @param paraIndex           The index of given instance.
	 * @return resultPrediction The prediction of given instance.
	 ********************
	 */
	public static double knn(int parakValue, Instances paraTraningSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure, int paraIndex) {
		int[] tempIndex = new int[parakValue + 2];
		double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(paraInstance);
		tempIndex = findNeighbor(parakValue, paraTraningSet, paraIndex, tempUnlabelInstanceValue, paraDistanceMeasure);
		double tempPrediction = 0;
		for (int i = 0; i < parakValue; i++) {
			tempPrediction += paraTraningSet.instance(tempIndex[i]).classValue();
		} // of for i
		tempPrediction = tempPrediction / parakValue;
		double resultPrediction = tempPrediction;
		return resultPrediction;
	}// of Knn

	/**
	 ********************
	 * Predict the value of the given instance.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraTraningSet      The training set of the regressor.
	 * @param paraInstance        The instance needs to find neighbor.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @param paraIndex1          The index of given instance.
	 * @param paraIndex2          The index of neighborhood.
	 * @return resultPrediction The prediction of given instance.
	 ********************
	 */
	public static double knn(int parakValue, Instances paraTraningSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure, int paraIndex1, int paraIndex2) {
		int[] tempIndex = new int[parakValue + 2];
		tempIndex = deNoiseFindNeighbor(parakValue, paraTraningSet, paraIndex1, paraIndex2, paraInstance,
				paraDistanceMeasure);
		double tempPrediction = 0;
		for (int i = 0; i < parakValue; i++) {
			tempPrediction += paraTraningSet.instance(tempIndex[i]).classValue();
		} // of for i
		tempPrediction = tempPrediction / parakValue;
		double resultPrediction = tempPrediction;
		return resultPrediction;
	}// of Knn

	/**
	 ************************* 
	 * Predict the value of the given instance.
	 * 
	 * 
	 * @param paraTraningSet The traningSet of the regressor.
	 * @param paraInstance   The given instance.
	 * @return resultPrediction The prediction of given instance.
	 ************************* 
	 */
	public double knn(Instances paraTraningSet, Instance paraInstance) {
		int[] tempIndex = new int[kValue + 2];
		double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(paraInstance);
		tempIndex = findNeighbor(kValue, paraTraningSet, tempUnlabelInstanceValue, distanceMeasure);
		double tempPrediction = 0;
		for (int i = 0; i < kValue; i++) {
			tempPrediction += paraTraningSet.instance(tempIndex[i]).classValue();
		}
		tempPrediction = tempPrediction / kValue;
		return tempPrediction;
	}// of Knn

	/**
	 ************************* 
	 * Predict the value of the given instance.
	 * 
	 * 
	 * @param paraTraningSet The traningSet of the regressor
	 * @param paraInstance   The given instance
	 * @param paraIndex      The index of given instance.
	 * @return resultPrediction The prediction of given instance.
	 ************************* 
	 */
	public double knn(Instances paraTraningSet, Instance paraInstance, int paraIndex) {
		int[] tempIndex = new int[kValue + 2];
		double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(paraInstance);
		tempIndex = findNeighbor(kValue, paraTraningSet, paraIndex, tempUnlabelInstanceValue, distanceMeasure);
		double tempPrediction = 0;
		for (int i = 0; i < kValue; i++) {
			tempPrediction += paraTraningSet.instance(tempIndex[i]).classValue();
		}
		tempPrediction = tempPrediction / kValue;
		double resultPrediction = tempPrediction;
		return resultPrediction;
	}// of Knn

	/**
	 * Get the prediction value of the given instance.
	 * 
	 * @return resultIndex The index of nearest neighborhood.
	 */
	public double[] regression() {
		double[] resultPrediction = new double[testingSet.numInstances()];
		for (int i = 0; i < testingSet.numInstances(); i++) {
			resultPrediction[i] = regression(trainingSet, testingSet.instance(i));
		}
		return resultPrediction;
	}

	/**
	 ************************* 
	 * Get the Prediction value of the given instance.
	 * 
	 * @param paraInstance The given instance
	 * @return The prediction of given instance.
	 ************************* 
	 */
	public double regression(Instance paraInstance) {
		return regression(trainingSet, paraInstance);
	}

	/**
	 ************************* 
	 * Get the Prediction value of the given instance.
	 * 
	 * @param paraInstance    The given instance.
	 * @param paraTrainingSet The training set.
	 * @return The prediction of given instance.
	 ************************* 
	 */
	public double regression(Instances paraTrainingSet, Instance paraInstance) {
		return knn(paraTrainingSet, paraInstance);
	}// of regression

	/**
	 ************************* 
	 * Predict the value of the given instance.
	 * 
	 * 
	 * @param paraTrainingSet The training set of the regressor.
	 * @param paraInstance    The given instance.
	 * @param paraIndex       The index of given instance.
	 * @return The prediction of given instance.
	 ************************* 
	 */
	public double regression(Instances paraTrainingSet, Instance paraInstance, int paraIndex) {
		return knn(paraTrainingSet, paraInstance, paraIndex);
	}// of regression

	/**
	 ********************
	 * Predict the value of the given instance.
	 * 
	 * @param paraTrainingSet The training set of the regressor.
	 * @param paraInstance    The instance needs to find neighbor.
	 * @param paraIndex1      The index of given instance.
	 * @param paraIndex2      The index of neighborhood.
	 * @return The prediction of given instance.
	 ********************
	 */
	public double regression(Instances paraTrainingSet, Instance paraInstance, int paraIndex1, int paraIndex2) {
		return knn(kValue, paraTrainingSet, paraInstance, distanceMeasure, paraIndex1, paraIndex2);
	}// of regression

	/**
	 ************************* 
	 * Get the Prediction value of the given instance.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraInstance        The given instance.
	 * @param paraTrainingSet     The training set.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @return The prediction of given instance.
	 ************************* 
	 */
	public static double regression(int parakValue, Instances paraTrainingSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure) {
		return knn(parakValue, paraTrainingSet, paraInstance, paraDistanceMeasure);
	}// of regression

	/**
	 ************************* 
	 * Get the Prediction value of the given instances.
	 * 
	 * @param parakValue          The kValue for kNNregressor.
	 * @param paraInstances       The given instances.
	 * @param paraTrainingSet     The training set.
	 * @param paraDistanceMeasure The distance measure (such as Manhattan, Euclidean distance) of the
	 *                            regressor.
	 * @return The prediction of given instance.
	 ************************* 
	 */
	public static double[] regression(int parakValue, Instances paraTrainingSet, Instances paraInstances,
			DistanceMeasure paraDistanceMeasure) {
		double[] tempPredictions = new double[paraInstances.numInstances()];
		for (int i = 0; i < paraInstances.numInstances(); i++) {
			tempPredictions[i] = regression(parakValue, paraTrainingSet, paraInstances.instance(i),
					paraDistanceMeasure);
		}
		return tempPredictions;
	}

	/**
	 ************************* 
	 * Compute the mean-squared error of the prediceted instance.
	 * 
	 * @param paraInstances  The given instances that will compute its mse.
	 * @param paraPrediction The prediction of the given instances.
	 * @return resultMse The mean squared error of given instances
	 ************************* 
	 */
	public double computeMse(Instances paraInstances, double[] paraPrediction) {
		double resultMse = 0;
		double tempMse = 0;
		double tempDifference = 0;
		for (int i = 0; i < paraPrediction.length; i++) {
			tempDifference = (paraInstances.instance(i).classValue() - paraPrediction[i]);
			tempMse += tempDifference * tempDifference;
		}
		resultMse = tempMse / paraInstances.numInstances();
		return resultMse;
	}// of computeMse

	/**
	 * Compute the mean-squared error of the testing instances.
	 * 
	 * @return resultMse The mean squared error of given instances
	 */
	public double computeMse() {
		double[] tempPrediction = regression();
		double resultMse = computeMse(testingSet, tempPrediction);
		return resultMse;
	}

	/**
	 ************************* 
	 * Test this class.
	 *
	 * @param args The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		Instances tempData = null;
		try {
			FileReader fileReader = new FileReader("src/data/kin8nm.arff");
			tempData = new Instances(fileReader);
			fileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: src/data/housing.arff.");
			System.exit(0);
		} // Of try
		tempData.setClassIndex(tempData.numAttributes() - 1);
		int k = 5;
		int[] tempTrainIdx = new int[(int) (tempData.numInstances() * 0.3)];
		for (int i = 0, j = 0; i < tempTrainIdx.length; i++) {
			tempTrainIdx[j] = i;
			j++;
		} // Of for i,j
		Instances tempTrainingSet = new Instances(tempData, 0);
		for (int i = 0; i < tempTrainIdx.length; i++)
			tempTrainingSet.add(tempData.instance(tempTrainIdx[i]));

		int[] tempTestIdx = new int[(int) (tempData.numInstances() * 0.3)];
		for (int i = (int) (tempData.numInstances() * (1 - 0.3)), j = 0; i < tempData.numInstances() - 1; i++) {
			tempTestIdx[j] = i;
			j++;
		} // Of for i,j
		Instances tempTestingSet = new Instances(tempData, 0);
		for (int i = 0; i < tempTestIdx.length; i++)
			tempTestingSet.add(tempData.instance(tempTestIdx[i]));

		KnnRegressor tempRegressor = new KnnRegressor(tempTrainingSet, tempTestingSet, DistanceMeasure.MAHALANOBIS, k);
		double tempMse = tempRegressor.computeMse();
		System.out.println("The test MSE is: " + tempMse);
	}// of main

}// of knnResgressor
