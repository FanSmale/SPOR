package common;

import java.io.FileReader;
import java.io.IOException;

import Jama.Matrix;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Distance measures.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive co-training project.
 * <p>
 * Progress: Three measures are implemented. More are desired.<br>
 * Written time: September 30, 2019. Copied from the active learning project. <br>
 * Last modify time: September 30, 2019.
 */

public class DistanceMeasure {
	/**
	 * The data. It should be modified in this class.
	 */
	public Instances data;

	/**
	 * The Euclidean distance.
	 */
	public static final int EUCLIDEAN = 0;

	/**
	 * The Manhattan distance.
	 */
	public static final int MANHATTAN = 1;

	/**
	 * The cosine distance.
	 */
	public static final int COSINE = 2;

	/**
	 * The MAHALANOBIS distance.
	 */
	public static final int MAHALANOBIS = 3;

	/**
	 * The current distance measure.
	 */
	int measure;

	/**
	 * The inverse means matrix for M... distance computation.
	 */
	Matrix inverseMeansMatrix;

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraData
	 *            The data set.
	 * @param paraMeasure
	 *            The measure.
	 ********************
	 */
	public DistanceMeasure(Instances paraData, int paraMeasure) {
		data = paraData;
		measure = paraMeasure;
		inverseMeansMatrix = null;
	}// Of the constructor

	/**
	 ************************* 
	 * Get the measure in int.
	 * 
	 * @return The measure in int.
	 ************************* 
	 */
	public int getMeasure() {
		return measure;
	}// Of getMeasure

	/**
	 ************************* 
	 * Compute the distance between two vectors.
	 * 
	 * @param paraFirstArray
	 *            The first array.
	 * @param paraSecondArray
	 *            The second array.
	 * @return The distance.
	 ************************* 
	 */
	public double distance(double[] paraFirstArray, double[] paraSecondArray) {
		double resultDistance = 0;
		switch (measure) {
		case EUCLIDEAN:
			resultDistance = euclideanDistance(paraFirstArray, paraSecondArray);
			break;
		case MANHATTAN:
			resultDistance = manhattanDistance(paraFirstArray, paraSecondArray);
			break;
		case COSINE:
			resultDistance = cosineDistance(paraFirstArray, paraSecondArray);
			break;
		case MAHALANOBIS:
			resultDistance = mahalanobisDistence(paraFirstArray, paraSecondArray);
			break;
		default:
			System.out.println("Unsupported distance measure: " + measure);
			System.exit(0);
		}// Of switch

		return resultDistance;
	}// Of distance

	/**
	 ************************* 
	 * Compute the distance between two instances.
	 * 
	 * @param paraFirstIndex
	 *            The first instance index.
	 * @param paraSecondIndex
	 *            The second instance index.
	 * @return The distance.
	 ************************* 
	 */
	public double distance(int paraFirstIndex, int paraSecondIndex) {
		double resultDistance = 0;
		switch (measure) {
		case EUCLIDEAN:
			resultDistance = euclideanDistance(paraFirstIndex, paraSecondIndex);
			break;
		case MANHATTAN:
			resultDistance = manhattanDistance(paraFirstIndex, paraSecondIndex);
			break;
		case COSINE:
			resultDistance = cosineDistance(paraFirstIndex, paraSecondIndex);
			break;
		case MAHALANOBIS:
			resultDistance = mahalanobisDistence(paraFirstIndex, paraSecondIndex);
			break;
		default:
			System.out.println("Unsupported distance measure: " + measure);
			System.exit(0);
		}// Of switch

		return resultDistance;
	}// Of distance

	/**
	 ************************* 
	 * Compute the distance between an instances and a vector.
	 * 
	 * @param paraIndex
	 *            The instance index.
	 * @param paraArray
	 *            The array.
	 * @return The distance.
	 ************************* 
	 */
	public double distance(int paraIndex, double[] paraArray) {
		double resultDistance = 0;
		switch (measure) {
		case EUCLIDEAN:
			resultDistance = euclideanDistance(paraIndex, paraArray);
			break;
		case MANHATTAN:
			resultDistance = manhattanDistance(paraIndex, paraArray);
			break;
		case COSINE:
			resultDistance = cosineDistance(paraIndex, paraArray);
			break;
		case MAHALANOBIS:
			resultDistance = mahalanobisDistence(paraIndex, paraArray);
			break;
		default:
			System.out.println("Unsupported distance measure: " + measure);
			System.exit(0);
		}// Of switch

		return resultDistance;
	}// Of distance

	/**
	 ************************* 
	 * Compute the Euclidean distance between two vectors.
	 * 
	 * @param paraFirstArray
	 *            The first array.
	 * @param paraSecondArray
	 *            The second array.
	 * @return The distance.
	 ************************* 
	 */
	public static double euclideanDistance(double[] paraFirstArray, double[] paraSecondArray) {
		double tempDifference = 0;
		double tempDistance = 0;

		for (int i = 0; i < paraFirstArray.length; i++) {
			tempDifference = paraFirstArray[i] - paraSecondArray[i];
			tempDistance += tempDifference * tempDifference;
			Common.runtimes++;
		} // Of for i

		return Math.sqrt(tempDistance);
	}// Of euclideanDistance

	/**
	 ************************* 
	 * Compute the Euclidean distance between two instances.
	 * 
	 * @param paraFirstIndex
	 *            The first instance index.
	 * @param paraSecondIndex
	 *            The second instance index.
	 * @return The distance.
	 ************************* 
	 */
	public double euclideanDistance(int paraFirstIndex, int paraSecondIndex) {
		double tempDifference = 0;
		double tempDistance = 0;

		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempDifference = data.instance(paraFirstIndex).value(i) - data.instance(paraSecondIndex).value(i);
			tempDistance += tempDifference * tempDifference;
			Common.runtimes++;
		} // Of for i

		return Math.sqrt(tempDistance);
	}// Of euclideanDistance

	/**
	 ************************* 
	 * Compute the Euclidean distance between an instances and a vector.
	 * 
	 * @param paraIndex
	 *            The instance index.
	 * @param paraArray
	 *            The array.
	 * @return The distance.
	 ************************* 
	 */
	public double euclideanDistance(int paraIndex, double[] paraArray) {
		double tempDifference = 0;
		double tempDistance = 0;

		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempDifference = data.instance(paraIndex).value(i) - paraArray[i];
			tempDistance += tempDifference * tempDifference;
			Common.runtimes++;
		} // Of for i

		return Math.sqrt(tempDistance);
	}// Of euclideanDistance

	/**
	 ************************* 
	 * Compute the Manhattan distance between two vectors.
	 * 
	 * @param paraFirstArray
	 *            The first array.
	 * @param paraSecondArray
	 *            The second array.
	 * @return The distance.
	 ************************* 
	 */
	public static double manhattanDistance(double[] paraFirstArray, double[] paraSecondArray) {
		double tempDifference = 0;
		double tempDistance = 0;

		for (int i = 0; i < paraFirstArray.length; i++) {
			tempDifference = paraFirstArray[i] - paraSecondArray[i];
			tempDistance += Math.abs(tempDifference);
		} // Of for i

		return tempDistance;
	}// Of manhattanDistance

	/**
	 ************************* 
	 * Compute the Manhattan distance between two instances.
	 * 
	 * @param paraFirstIndex
	 *            The first instance index.
	 * @param paraSecondIndex
	 *            The second instance index.
	 * @return The distance.
	 ************************* 
	 */
	public double manhattanDistance(int paraFirstIndex, int paraSecondIndex) {
		double tempDifference = 0;
		double tempDistance = 0;

		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempDifference = data.instance(paraFirstIndex).value(i) - data.instance(paraSecondIndex).value(i);
			tempDistance += Math.abs(tempDifference);
		} // Of for i

		return tempDistance;
	}// Of manhattanDistance

	/**
	 ************************* 
	 * Compute the Manhattan distance between an instances and a vector.
	 * 
	 * @param paraIndex
	 *            The instance index.
	 * @param paraArray
	 *            The array.
	 * @return The distance.
	 ************************* 
	 */
	public double manhattanDistance(int paraIndex, double[] paraArray) {
		double tempDifference = 0;
		double tempDistance = 0;

		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempDifference = data.instance(paraIndex).value(i) - paraArray[i];
			tempDistance += Math.abs(tempDifference);
		} // Of for i

		return tempDistance;
	}// Of manhattanDistance

	/**
	 ************************* 
	 * Compute the cosine distance between two vectors.
	 * 
	 * @param paraFirstArray
	 *            The first array.
	 * @param paraSecondArray
	 *            The second array.
	 * @return The distance.
	 ************************* 
	 */
	public static double cosineDistance(double[] paraFirstArray, double[] paraSecondArray) {
		double tempDistance = 0;
		double tempNumerator = 0;
		double tempRecordX = 0;
		double tempRecordY = 0;
		double tempDenominator = 0;

		for (int i = 0; i < paraFirstArray.length; i++) {
			tempNumerator += paraFirstArray[i] * paraSecondArray[i];
			tempRecordX += Math.pow(paraFirstArray[i], 2);
			tempRecordY += Math.pow(paraSecondArray[i], 2);
		} // Of for i
		tempDenominator = Math.sqrt(tempRecordX) * Math.sqrt(tempRecordY);
		tempDistance = tempNumerator / tempDenominator;
		return tempDistance;
	}// Of cosineDistance

	/**
	 ************************* 
	 * Compute the cosine distance between two instances.
	 * 
	 * @param paraFirstIndex
	 *            The first instance index.
	 * @param paraSecondIndex
	 *            The second instance index.
	 * @return The distance.
	 ************************* 
	 */
	public double cosineDistance(int paraFirstIndex, int paraSecondIndex) {
		double tempDistance = 0;
		double[] paraOneIndexArray = new double[data.numAttributes() - 1];
		double[] paraTwoIndexArray = new double[data.numAttributes() - 1];
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			paraOneIndexArray[i] = data.instance(paraFirstIndex).value(i);
			paraTwoIndexArray[i] = data.instance(paraSecondIndex).value(i);
		} // Of for i
		tempDistance = cosineDistance(paraOneIndexArray, paraTwoIndexArray);
		return tempDistance;
	}// Of cosineDistance

	/**
	 ************************* 
	 * Compute the cosine distance between an instances and a vector.
	 * 
	 * @param paraIndex
	 *            The instance index.
	 * @param paraArray
	 *            The array.
	 * @return The distance.
	 ************************* 
	 */
	public double cosineDistance(int paraIndex, double[] paraArray) {
		double tempDistance = 0;
		double[] tempArray = new double[data.numAttributes() - 1];
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempArray[i] = data.instance(paraIndex).value(i);
		} // Of for i
		tempDistance = cosineDistance(tempArray, paraArray);
		return tempDistance;
	}// Of cosineDistance

	/**
	 ************************* 
	 * Compute the Mahalanobis distance between two vectors.
	 * 
	 * @param paraFirstArray
	 *            The first array.
	 * @param paraSecondArray
	 *            The second array.
	 ************************* 
	 */
	double mahalanobisDistence(double[] paraFirstArray, double[] paraSecondArray) {
		if (inverseMeansMatrix == null) {
			// Initialize it
			double[] tempMeans = new double[data.numAttributes() - 1];
			Matrix oriData = new Matrix(data.numInstances(), data.numAttributes() - 1);

			for (int i = 0; i < data.numInstances(); i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempMeans[j] += data.instance(i).value(j);
					oriData.set(i, j, data.instance(i).value(j));
				} // Of for j
			} // Of for i

			for (int j = 0; j < data.numAttributes() - 1; j++) {
				tempMeans[j] /= data.numInstances();
			} // Of for j

			for (int i = 0; i < data.numInstances(); i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					oriData.set(i, j, oriData.get(i, j) - tempMeans[j]);
				} // Of for j
			} // Of for i

			Matrix meansMartrix = oriData;

			Matrix covarianceMatrix = (meansMartrix.transpose()).times(meansMartrix);

			inverseMeansMatrix = covarianceMatrix.inverse();

		} // Of if

		double[][] tempMar1 = new double[1][paraFirstArray.length];
		double[][] tempMar2 = new double[1][paraSecondArray.length];
		for (int i = 0; i < paraFirstArray.length; i++) {
			tempMar1[0][i] = paraFirstArray[i];
			tempMar2[0][i] = paraSecondArray[i];
		}

		Matrix tempMatrix1 = new Matrix(tempMar1);
		Matrix tempMatrix2 = new Matrix(tempMar2);

		Matrix tempDifference = tempMatrix1.minus(tempMatrix2);

		Matrix tempMatrixResult = ((tempDifference).times(inverseMeansMatrix)).times(tempDifference.transpose());

		double result = Math.sqrt(tempMatrixResult.get(0, 0));

		// System.out.println("Distance between " +
		// Arrays.toString(paraFirstArray) + " and "
		// + Arrays.toString(paraSecondArray) + " is " + result);
		return result;
	}// Of mahalanobisDistence

	/**
	 ************************* 
	 * Compute the Mahalanobis distance between an instances and a vector.
	 * 
	 * @param paraIndex
	 *            The instance index.
	 * @param paraArray
	 *            The array.
	 ************************* 
	 */
	double mahalanobisDistence(int paraIndex, double[] paraSecondArray) {
		double result = 0;
		double[] paraIndexArray = new double[paraSecondArray.length];
		for (int i = 0; i < paraSecondArray.length; i++) {
			paraIndexArray[i] = data.instance(paraIndex).value(i);
		} // Of for i

		result = mahalanobisDistence(paraIndexArray, paraSecondArray);
		return result;
	}// Of mahalanobisDistence

	/**
	 ************************* 
	 * Compute the Mahalanobis distance between two instances.
	 * 
	 * @param paraFirstIndex
	 *            The first instance index.
	 * @param paraSceondIndex
	 *            The second instance index.
	 ************************* 
	 */
	double mahalanobisDistence(int paraIndex1, int paraIndex2) {
		double result = 0;
		double[] tempFirstArray = new double[data.numAttributes() - 1];
		double[] tempSecondArray = new double[data.numAttributes() - 1];
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempFirstArray[i] = data.instance(paraIndex1).value(i);
			tempSecondArray[i] = data.instance(paraIndex2).value(i);
		} // Of for i

		result = mahalanobisDistence(tempFirstArray, tempSecondArray);
		return result;
	}// Of mahalanobisDistence
	
	/**
	 ************************* 
	 * An instance converted to a double array, where the class label is not considered.
	 * 
	 * @param paraInstance
	 *            The given instance.
	 * @return A double array.
	 ************************* 
	 */
	public static double[] instanceToDoubleArray(Instance paraInstance) {
		double[] resultArray = new double[paraInstance.numAttributes() - 1];
		for (int i = 0; i < resultArray.length; i++) {
			resultArray[i] = paraInstance.value(i);
		}//Of for i
		
		return resultArray;
	}//Of instanceToDoubleArray

	/**
	 ************************* 
	 * An instance converted to a double array, where the class label is not considered.
	 * 
	 * @param paraInstance
	 *            The given instance.
	 *            @param paraFeatureSubset The given feature subset.
	 * @return A double array.
	 ************************* 
	 */
	public static double[] instanceToDoubleArray(Instance paraInstance, int[] paraFeatureSubset) {
		double[] resultArray = new double[paraFeatureSubset.length];
		for (int i = 0; i < resultArray.length; i++) {
			resultArray[i] = paraInstance.value(paraFeatureSubset[i]);
		}//Of for i
		
		return resultArray;
	}//Of instanceToDoubleArray

	/**
	 ************************* 
	 * Display the distance measure.
	 * 
	 * @return The distance measure.
	 ************************* 
	 */
	public String toString() {
		String resultString = null;

		switch (measure) {
		case EUCLIDEAN:
			resultString = "Euclidean";
			break;
		case MANHATTAN:
			resultString = "Manhattan";
			break;
		case COSINE:
			resultString = "Cosine";
			break;
		case MAHALANOBIS:
			resultString = "MahalanobisDistence";
			break;
		default:
			System.out.println("Unsupported distance measure: " + measure);
			System.exit(0);
		}// Of switch

		return resultString;
	}// Of toString

	/**
	 ************************* 
	 * The test entrance
	 * 
	 * @author Fan Min
	 * @param args
	 *            The parameters.
	 * @throws IOException
	 *             The IOException for data reading.
	 ************************* 
	 */
	public static void main(String[] args) throws IOException {
		System.out.println("Hello.");
		FileReader fileReader = new FileReader("src/data/iris.arff");
		Instances data1 = new Instances(fileReader);
		fileReader.close();

		double[] tempDataIndex0 = { 5.1, 3.5, 1.4, 0.2 };
		double[] tempData2Index1 = { 5.1, 3.5, 1.4, 0.2 };

		DistanceMeasure tempMeasure = new DistanceMeasure(data1, COSINE);
		double tempDistance1 = tempMeasure.distance(0, 1);
		System.out.println("Distance between two instances: " + tempDistance1);

		double tempDistance2 = tempMeasure.distance(0, tempData2Index1);
		System.out.println("Distance between an instance and an array: " + tempDistance2);

		double tempDistance3 = tempMeasure.distance(tempDataIndex0, tempData2Index1);
		System.out.println("Distance between two arrays: " + tempDistance3);

	}// Of main
}// Of class Distance
