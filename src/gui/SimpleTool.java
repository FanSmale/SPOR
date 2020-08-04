package gui;

import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Random;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
//import algorithm.IBkReg.NeighborNode;
//import algorithm.IBkReg.NeighborList;;

public class SimpleTool {
	public static long runtime = 0;
	
	public static long NumInstances1added=0;
	public static long NumInstances2added=0;
	public static double errorDrop=0;
	public static double maxErrorDrop=0;
	public static double minErrorDrop=Double.MAX_VALUE;

	/**
	 ************************************** 
	 * Perform min-max normalization. Normalize to [0, 1]. The data table is changed
	 * directly.
	 * 
	 * @param paraData : the data set.
	 * 
	 ************************************** 
	 */
	public static void normalize(Instances paraData) {
		for (int i = 0; i < paraData.numAttributes(); i++) {
			double[] tempVals = new double[paraData.numInstances()];
			double max = Double.MIN_VALUE;
			double min = Double.MAX_VALUE;
			for (int j = 0; j < paraData.numInstances(); j++) {
				tempVals[j] = paraData.instance(j).value(i);

				if (max < tempVals[j]) {
					max = tempVals[j];
				} // Of if

				if (min > tempVals[j]) {
					min = tempVals[j];
				} // Of if

			} // Of for j

			for (int j = 0; j < paraData.numInstances(); j++) {
				double newval = (tempVals[j] - min) / (max - min);
				paraData.instance(j).setValue(i, newval);
			} // Of for j
		} // Of for i
	}// Of normalize

	/**
	 ********************************** 
	 * Get a random order index array.
	 * 
	 * @param paraLength The length of the array.
	 * @return A random order.
	 ********************************** 
	 */
	public static int[] getRandomOrder(int paraLength) {
		// Step 1. Initialize
		int[] resultArray = new int[paraLength];
		for (int i = 0; i < paraLength; i++) {
			resultArray[i] = i;
		} // Of for i

		// Step 2. Swap many times
		Random random = new Random();
		int tempFirst, tempSecond;
		int tempValue;
		for (int i = 0; i < paraLength * 10; i++) {
			tempFirst = random.nextInt(paraLength);
			tempSecond = random.nextInt(paraLength);

			tempValue = resultArray[tempFirst];
			resultArray[tempFirst] = resultArray[tempSecond];
			resultArray[tempSecond] = tempValue;
		} // Of for i

		return resultArray;
	}// Of getRandomOrder

	/**
	 ********************************** 
	 * reverse a the paraBlock
	 * 
	 * @param parablock The given Block.
	 ********************************** 
	 */
	public static int[] reverse(int[] paraBlock) {
		for (int i = 0; i < paraBlock.length / 2; i++) {
			int tem = paraBlock[i];
			paraBlock[i] = paraBlock[paraBlock.length - 1 - i];
			paraBlock[paraBlock.length - 1 - i] = tem;
		} // Of for i
		return paraBlock;
	}// Of reverse

	/**
	 ********************************** 
	 * Disorder a dataset, so that the order does not influence the results.
	 * 
	 * @param paraFilename The given filename.
	 ********************************** 
	 */
	public static void disorderData(String paraFilename) {
		// Step 1. Read the data.
		Instances tempData = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			tempData = new Instances(fileReader);
			fileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try

		// Step 2. Disorder.
		int[] tempNewOrders = getRandomOrder(tempData.numInstances());
		// Copy.
		Instances tempNewData = new Instances(tempData);
		tempNewData.delete();
		System.out.println("The empty data is: " + tempNewData);
		for (int i = 0; i < tempNewOrders.length; i++) {
			tempNewData.add(tempData.instance(tempNewOrders[i]));
		} // Of for i

		System.out.println("Writing to a new file:");
		// Step 3. Write to a new file.
		int tempLength = paraFilename.length();
		String tempNewFilename = paraFilename.substring(0, tempLength - 5) + "_disorder.arff";
		System.out.println(tempNewFilename);
		try {
			RandomAccessFile tempNewFile = new RandomAccessFile(tempNewFilename, "rw");
			tempNewFile.writeBytes(tempNewData.toString());
			tempNewFile.close();
		} catch (IOException ee) {
			System.out.println(ee);
		} // Of try
	}// Of disorderData

	/**
	 ********************
	 * judge two given instance is the same instance
	 *
	 * @param paraFirstInstance  the first given instance
	 * @param paraSecondInstance the second given instance
	 * @return true or false
	 ********************
	 */
	public static boolean equal(Instance paraFirstInstance, Instance paraSecondInstance) {
		for (int i = 0; i < paraFirstInstance.numAttributes() - 1; i++) {
			if (paraFirstInstance.value(i) != paraSecondInstance.value(i)) {
				return false;
			} // Of for i
		}
		return true;
	}// Of equal

	/**
	 ********************
	 * Computes the difference between two given attribute values.
	 * 
	 * @param paraIndex the index number of given instance
	 * @param tempval1  the first given given attribute values
	 * @param tempval2  the second given given attribute values
	 ********************
	 */
	protected static double difference(int paraIndex, double paraValue1, double paraValue2, Instances trainSet,
			boolean dontNormalize, double[] minInstance, double[] maxInstance) {
		runtime++;
		switch (trainSet.attribute(paraIndex).type()) {
		case Attribute.NOMINAL:

			// If attribute is nominal
			if (Instance.isMissingValue(paraValue1) || Instance.isMissingValue(paraValue2)
					|| ((int) paraValue1 != (int) paraValue2)) {
				return 1;
			} else {
				return 0;
			} // Of if
		case Attribute.NUMERIC:

			// If attribute is numeric
			if (Instance.isMissingValue(paraValue1) || Instance.isMissingValue(paraValue2)) {
				if (Instance.isMissingValue(paraValue1) && Instance.isMissingValue(paraValue2)) {
					return 1;
				} else {
					double tempDifference;
					if (Instance.isMissingValue(paraValue2)) {
						tempDifference = norm(paraValue1, paraIndex, dontNormalize, minInstance, maxInstance);
					} else {
						tempDifference = norm(paraValue2, paraIndex, dontNormalize, minInstance, maxInstance);
					} // Of if
					if (tempDifference < 0.5) {
						tempDifference = 1.0 - tempDifference;
					} // Of if
					return tempDifference;
				}
			} else {
				return norm(paraValue1, paraIndex, dontNormalize, minInstance, maxInstance)
						- norm(paraValue2, paraIndex, dontNormalize, minInstance, maxInstance);
			} // Of if
		default:
			return 0;
		}// Of switch
	}// Of difference

	/**
	 ********************
	 * Normalizes a given value of a numeric attribute.
	 *
	 * @param paraValue1 the value to be normalized
	 * @param paraIndex, the attribute's index
	 ********************
	 */
	public static double norm(double paraValue1, int paraIndex, boolean dontNormalize, double[] minInstance,
			double[] maxInstance) {

		if (dontNormalize) {
			return paraValue1;
		} else if (Double.isNaN(minInstance[paraIndex]) || Utils.eq(maxInstance[paraIndex], minInstance[paraIndex])) {
			return 0;
		} else {
			return (paraValue1 - minInstance[paraIndex]) / (maxInstance[paraIndex] - minInstance[paraIndex]);
		} // Of if
	}// Of norm

	/**
	 ********************
	 * Calculates the Distance between two instances
	 *
	 * @param paraFirst     The first instance
	 * @param paraSecond    The second instance
	 * @param trainSet      The set of training instances
	 * @param dontNormalize The number whether normalization is turned off
	 * @param minInstance   The minimum values for numeric attributes.
	 * @param maxInstance   The maximum values for numeric attributes.
	 * @param metricOrder   Possible instance weighting methods
	 * @param invConvMatrix The ConvMatrix used in Mahalanobis distance
	 * @param metricFlag    The distance measure.
	 * @return the paraDistance between the two given instances, between 0 and 1
	 ********************
	 */
	public static double distance(Instance paraFirst, Instance paraSecond, Instances trainSet, boolean dontNormalize,
			double[] minInstance, double[] maxInstance, double metricOrder, double[][] invCovMatrix,
			String metricFlag) {

		if (metricFlag.compareTo("M") == 0) {
			return distance_M(paraFirst, paraSecond, invCovMatrix);
		} else {
			return distance_E(paraFirst, paraSecond, trainSet, dontNormalize, minInstance, maxInstance, metricOrder);
		} // Of if
	}// Of paraDistance

	/**
	 ********************
	 * Calculates the Mahalanobis distance between two instances
	 *
	 * @param paraFirst     the first instance
	 * @param paraSecond    the second instance
	 * @param invConvMatrix The ConvMatrix used in Mahalanobis distance
	 * @return the Mahalanobis distance between the two given instances, between 0
	 *         and 1
	 ********************
	 */

	public static double distance_M(Instance paraFirst, Instance paraSecond, double[][] invCovMatrix) {
		
		double tempDistance = 0;
		double[] temp = new double[paraFirst.numAttributes() - 1];
		double[] tempDifference = new double[paraFirst.numAttributes() - 1];
		for (int i = 0; i < temp.length; i++) {
			tempDifference[i] = paraFirst.value(i) - paraSecond.value(i);
		} // Of for i

		for (int j = 0; j < temp.length; j++) {
			for (int i = 0; i < temp.length; i++) {
				temp[j] += tempDifference[i] * invCovMatrix[i][j];
			} // Of for i
		} // Of for j

		for (int j = 0; j < temp.length; j++) {
			tempDistance += temp[j] * tempDifference[j];
		} // Of for j

		return tempDistance;
	}// Of distance_M

	/**
	 ********************
	 * Calculates the Euclidean distance between two instances
	 *
	 * @param paraFirst     the first instance
	 * @param paraSecond    the second instance
	 * @param trainSet      The set of training instances
	 * @param dontNormalize The number whether normalization is turned off
	 * @param minInstance   The minimum values for numeric attributes.
	 * @param maxInstance   The maximum values for numeric attributes.
	 * @param metricOrder   Possible instance weighting methods
	 * @param invConvMatrix The ConvMatrix used in Mahalanobis distance
	 * @return the Euclidean distance between the two given instances, between 0 and
	 *         1
	 ********************
	 */

	public static double distance_E(Instance paraFirst, Instance paraSecond, Instances trainSet, boolean dontNormalize,
			double[] minInstance, double[] maxInstance, double metricOrder) {
		double tempDistance = 0;
		int tempFirst, tempSecond;
		
		for (int i = 0, j = 0; i < paraFirst.numValues() || j < paraSecond.numValues();) {
			if (i >= paraFirst.numValues()) {
				tempFirst = trainSet.numAttributes();
			} else {
				tempFirst = paraFirst.index(i);
			} // Of if
			if (j >= paraSecond.numValues()) {
				tempSecond = trainSet.numAttributes();
			} else {
				tempSecond = paraSecond.index(j);
			} // Of if
			if (tempFirst == trainSet.classIndex()) {
				i++;
				continue;
			} // Of if
			if (tempSecond == trainSet.classIndex()) {
				j++;
				continue;
			} // Of if
			double tempDifference;
			if (tempFirst == tempSecond) {
				tempDifference = SimpleTool.difference(tempFirst, paraFirst.valueSparse(i), paraSecond.valueSparse(j),
						trainSet, dontNormalize, minInstance, maxInstance);
				i++;
				j++;
			} else if (tempFirst > tempSecond) {
				tempDifference = SimpleTool.difference(tempFirst, paraFirst.valueSparse(i), paraSecond.valueSparse(j),
						trainSet, dontNormalize, minInstance, maxInstance);
				j++;
			} else {
				tempDifference = SimpleTool.difference(tempFirst, paraFirst.valueSparse(i), paraSecond.valueSparse(j),
						trainSet, dontNormalize, minInstance, maxInstance);
				i++;
			} // Of if
			tempDistance += Math.pow(Math.abs(tempDifference), metricOrder);
		} // Of for p1,p2

		return Math.pow(tempDistance, (1 / metricOrder));
	}// Of distance_E
}
