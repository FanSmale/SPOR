package regressor;

import java.io.FileReader;
import java.util.Arrays;
import common.*;
import weka.core.*;

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
	 * The distanceMeasure value for regressor.
	 */
	DistanceMeasure distanceMeasure;

	/**
	 * Get the distanceMeasure value for regressor.
	 */
	public DistanceMeasure getDistanceMeasure() {
		return distanceMeasure;
	}// of getDistanceMeasure

	/**
	 * Set the distanceMeasure value for regressor.
	 */
	public void setDistanceMeasure(Instances paraTrainingSet, int paraDistanceMeasure) {
		distanceMeasure = new DistanceMeasure(paraTrainingSet, paraDistanceMeasure);
	}// of setDistanceMeasure

	/**
	 * Get the k value for kNNregressor.
	 */
	public int getkValue() {
		return kValue;
	}// of getkValue

	/**
	 * Set the k value for kNNregressor.
	 * 
	 * @param The k value for kNNregressor
	 */
	public void setKvalue(int paraKvalue) {
		kValue = paraKvalue;
	}// of setkValue

	/**
	 * Update the traningSet.
	 * 
	 * @param paraInstances The newly traningSet
	 */
	public void updatetrainingSet(Instances paraInstances) {
		trainingSet = paraInstances;
	}

	/**
	 * Build the regressor.
	 */
	public KnnRegressor(Instances paraTraningSet, Instances paraTestingSet) {
		trainingSet = paraTraningSet;
		testingSet = paraTestingSet;
	}// of constructor

	/**
	 * Build the regressor.
	 */
	public KnnRegressor(Instances paraTraningSet, Instances paraTestingSet, int paraDistanceMeasure, int paraKvalue) {
		this(paraTraningSet, paraTestingSet);
		setDistanceMeasure(trainingSet, paraDistanceMeasure);
		setKvalue(paraKvalue);
	}// of constructor

	public KnnRegressor(Instances paraTraningSet, Instances paraTestingSet, DistanceMeasure paraDistanceMeasure,
			int paraKvalue) {
		this(paraTraningSet, paraTestingSet);
		distanceMeasure = paraDistanceMeasure;
		setKvalue(paraKvalue);
	}// of constructor

	/**
	 ********************
	 * Find the neighbor of the given instance.
	 * 
	 * @param paraKvalue        The k value for kNNregressor
	 * @param paraTraningSet    The traningSet of the regressor
	 * @param paraUnlabeleddata The value of the given instance
	 ********************
	 */
	public static int[] findNeighbor(int paraKvalue, Instances paraTraningSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure) {
		double[] tempUnlabeleddata = DistanceMeasure.instanceToDoubleArray(paraInstance);
		int[] tempIndex = new int[paraKvalue + 2];
		int[] resultIndex = new int[paraKvalue];
		double[] tempDistance = new double[paraKvalue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(tempUnlabeleddata, tempTraningValue);
			for (int j = paraKvalue;; j--) {
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
		for (int i = 0; i < paraKvalue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	public static int[] findNeighbor(int paraKvalue, Instances paraTraningSet, int paraIndex, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure) {
		double[] tempUnlabeleddata = DistanceMeasure.instanceToDoubleArray(paraInstance);
		int[] tempIndex = new int[paraKvalue + 2];
		int[] resultIndex = new int[paraKvalue];
		double[] tempDistance = new double[paraKvalue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			if (i == paraIndex) {
				continue;
			} // of if
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(tempUnlabeleddata, tempTraningValue);
			for (int j = paraKvalue;; j--) {
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
		for (int i = 0; i < paraKvalue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	public static int[] findNeighbor(int paraKvalue, Instances paraTraningSet, double[] paraUnlabeleddata,
			DistanceMeasure paraDistanceMeasure) {
		int[] tempIndex = new int[paraKvalue + 2];
		int[] resultIndex = new int[paraKvalue];
		double[] tempDistance = new double[paraKvalue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(paraUnlabeleddata, tempTraningValue);
			for (int j = paraKvalue;; j--) {
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
		for (int i = 0; i < paraKvalue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	public static int[] findNeighbor(int paraKvalue, Instances paraTraningSet, int paraIndex,
			double[] paraUnlabeleddata, DistanceMeasure paraDistanceMeasure) {
		int[] tempIndex = new int[paraKvalue + 2];
		int[] resultIndex = new int[paraKvalue];
		double[] tempDistance = new double[paraKvalue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			if (i == paraIndex) {
				continue;
			} // of if
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(paraUnlabeleddata, tempTraningValue);
			for (int j = paraKvalue;; j--) {
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
		for (int i = 0; i < paraKvalue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	public static int[] deNoiseFindNeighbor(int paraKvalue, Instances paraTraningSet, int paraIndex1, int paraIndex2,
			Instance paraInstance, DistanceMeasure paraDistanceMeasure) {
		double[] tempUnlabeleddata = DistanceMeasure.instanceToDoubleArray(paraInstance);
		int[] tempIndex = new int[paraKvalue + 2];
		int[] resultIndex = new int[paraKvalue];
		double[] tempDistance = new double[paraKvalue + 2];
		Arrays.fill(tempIndex, -1);
		Arrays.fill(tempDistance, Double.MAX_VALUE);
		tempDistance[0] = -1;
		double tempCurrentDistance;
		double[] tempTraningValue;
		for (int i = 0; i < paraTraningSet.numInstances(); i++) {
			if (i == paraIndex1 || i == paraIndex2) {
				continue;
			}
			tempTraningValue = DistanceMeasure.instanceToDoubleArray(paraTraningSet.instance(i));
			tempCurrentDistance = paraDistanceMeasure.distance(tempUnlabeleddata, tempTraningValue);
			for (int j = paraKvalue;; j--) {
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
		for (int i = 0; i < paraKvalue; i++) {
			resultIndex[i] = tempIndex[i + 1];
		} // of for i
		return resultIndex;
	}// of findNeighbor

	/**
	 ********************
	 * Predict the value of the given instance.
	 * 
	 * @param paraTraningSet The traningSet of the regressor
	 * @param paraInstance   The given instance
	 ********************
	 */
	public static double knn(int paraKvalue, Instances paraTraningSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure) {
		int[] tempIndex = new int[paraKvalue + 2];
		double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(paraInstance);
		tempIndex = findNeighbor(paraKvalue, paraTraningSet, tempUnlabelInstanceValue, paraDistanceMeasure);
		double tempPrediction = 0;
		for (int i = 0; i < paraKvalue; i++) {
			tempPrediction += paraTraningSet.instance(tempIndex[i]).classValue();
		}// of for i
		tempPrediction = tempPrediction / paraKvalue;
		return tempPrediction;
	}// of Knn

	public static double knn(int paraKvalue, Instances paraTraningSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure, int paraIndex) {
		int[] tempIndex = new int[paraKvalue + 2];
		double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(paraInstance);
		tempIndex = findNeighbor(paraKvalue, paraTraningSet, paraIndex, tempUnlabelInstanceValue, paraDistanceMeasure);
		double tempPrediction = 0;
		for (int i = 0; i < paraKvalue; i++) {
			tempPrediction += paraTraningSet.instance(tempIndex[i]).classValue();
		}// of for i 
		tempPrediction = tempPrediction / paraKvalue;
		return tempPrediction;
	}// of Knn

	public static double knn(int paraKvalue, Instances paraTraningSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure, int paraIndex1, int paraIndex2) {
		int[] tempIndex = new int[paraKvalue + 2];
		tempIndex = deNoiseFindNeighbor(paraKvalue, paraTraningSet, paraIndex1, paraIndex2, paraInstance,
				paraDistanceMeasure);
		double tempPrediction = 0;
		for (int i = 0; i < paraKvalue; i++) {
			tempPrediction += paraTraningSet.instance(tempIndex[i]).classValue();
		}// of for i
		tempPrediction = tempPrediction / paraKvalue;
		return tempPrediction;
	}// of Knn

	/**
	 ************************* 
	 * Predict the value of the given instance.
	 * 
	 * 
	 * @param paraTraningSet The traningSet of the regressor
	 * @param paraInstance   The given instance
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

	public double knn(Instances paraTraningSet, Instance paraInstance, int paraIndex) {
		int[] tempIndex = new int[kValue + 2];
		double[] tempUnlabelInstanceValue = DistanceMeasure.instanceToDoubleArray(paraInstance);
		tempIndex = findNeighbor(kValue, paraTraningSet, paraIndex, tempUnlabelInstanceValue, distanceMeasure);
		double tempPrediction = 0;
		for (int i = 0; i < kValue; i++) {
			tempPrediction += paraTraningSet.instance(tempIndex[i]).classValue();
		}
		tempPrediction = tempPrediction / kValue;
		return tempPrediction;
	}// of Knn

	/**
	 * Get the prediction value of the given instance.
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
	 ************************* 
	 */
	public double regression(Instance paraInstance) {
		return regression(trainingSet, paraInstance);
	}
	public double regression(Instances paraTrainingSet, Instance paraInstance) {
		return knn(paraTrainingSet, paraInstance);
	}// of regression

	public double regression(Instances paraTrainingSet, Instance paraInstance, int paraIndex) {
		return knn(paraTrainingSet, paraInstance, paraIndex);
	}// of regression

	public double regression(Instances paraTrainingSet, Instance paraInstance, int paraIndex1, int paraIndex2) {
		return knn(kValue, paraTrainingSet, paraInstance, distanceMeasure, paraIndex1, paraIndex2);
	}// of regression

	public static double regression(int paraKvalue, Instances paraTraningSet, Instance paraInstance,
			DistanceMeasure paraDistanceMeasure) {
		return knn(paraKvalue, paraTraningSet, paraInstance, paraDistanceMeasure);
	}// of regression

	public static double[] regression(int paraKvalue, Instances paraTraningSet, Instances paraInstances,
			DistanceMeasure paraDistanceMeasure) {
		double[] tempPredictions = new double[paraInstances.numInstances()];
		for (int i = 0; i < paraInstances.numInstances(); i++) {
			tempPredictions[i] = regression(paraKvalue, paraTraningSet, paraInstances.instance(i), paraDistanceMeasure);
		}
		return tempPredictions;
	}

	/**
	 ************************* 
	 * Compute the mean-squared error of the prediceted instance.
	 * 
	 * @param paraInstances  The given instances that will compute its mse
	 * @param paraPrediction The prediction of the given instances
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
