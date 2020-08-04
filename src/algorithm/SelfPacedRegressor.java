package algorithm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import weka.core.Instances;

import Jama.*;
import gui.SimpleTool;
import common.SimpleTools;

/**
 * Self-paced regression. <br>
 * Project: Self-paced learning.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/MFAdaBoosting.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 *         Data Created: July 26, 2020.<br>
 *         Last modified: July 26, 2020.
 * @version 1.0
 */

public class SelfPacedRegressor {

	/**
	 * The whole input data. The first column is always 1.
	 */
	double[][] wholeX;

	/**
	 * The whole output data.
	 */
	double[][] wholeY;

	/**
	 * The training data input.
	 */
	double[][] trainingX;

	/**
	 * The training data output.
	 */
	double[][] trainingY;

	/**
	 * The testing data input.
	 */
	double[][] testingX;

	/**
	 * The testing data output.
	 */
	double[][] testingY;

	/**
	 * The weights for the training hyper-space.
	 */
	double[] weights;

	/**
	 * The initial distance threshold.
	 */
	double distanceThresholdInitial = 0.2;

	/**
	 * The incremental distance threshold.
	 */
	double distanceThresholdIncrement = 0.2;
	public static double tempValue=0;
	public static double tempThreshold=0;
	public static int  beforeinstance=0;
	public static double afterinstance=0;
	/**
	 ****************** 
	 * The first constructor.
	 * 
	 * @param paraTrainingFilename
	 *            The data filename.
	 ****************** 
	 */
	public SelfPacedRegressor(String paraTrainingFilename) {
		Instances data = null;
		// Step 1. Read training set.
		try {
			FileReader tempFileReader = new FileReader(paraTrainingFilename);
			data = new Instances(tempFileReader);
			tempFileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraTrainingFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try
		SimpleTool.normalize(data);
		int tempNumInstances = data.numInstances();
		int tempNumAttributes = data.numAttributes();

		wholeX = new double[tempNumInstances][tempNumAttributes];
		wholeY = new double[tempNumInstances][1];

		for (int i = 0; i < tempNumInstances; i++) {
			// The first element is always set to 1.
			wholeX[i][0] = 1;
			wholeY[i][0] = data.instance(i).value(tempNumAttributes - 1);
			for (int j = 0; j < tempNumAttributes - 1; j++) {
				wholeX[i][j + 1] = data.instance(i).value(j);
			} // Of for j
		} // Of for i

		//System.out.println("The input space is: " + Arrays.deepToString(wholeX));
		//System.out.println("The output space is: " + Arrays.toString(wholeY));
	}// Of the first constructor

	/**
	 ****************** 
	 * Setter.
	 * 
	 * @param paraInitial
	 *            The initial value.
	 * @param paraIncrement
	 *            The increment value.
	 ****************** 
	 */
	public void setDistanceThresholds(double paraInitial, double paraIncrement) {
		distanceThresholdInitial = paraInitial;
		distanceThresholdIncrement = paraIncrement;
	}// Of setDistanceThresholds

	/**
	 ****************** 
	 * Randomize the training and testing sets.
	 * 
	 * @param paraTrainingFraction
	 *            The fraction of the training set.
	 ****************** 
	 */
	public void randomizeTrainingTesting(double paraTrainingFraction) {
		// Step 1. Randomize a sequence.
		int[] tempSequence = SimpleTools.getRandomOrder(wholeY.length);

		// Step 2. Copy the training set.
		int tempTrainingSize = (int) (wholeY.length * paraTrainingFraction);
		// For X, only space for references.
		trainingX = new double[tempTrainingSize][];
		trainingY = new double[tempTrainingSize][];
		for (int i = 0; i < tempTrainingSize; i++) {
			trainingX[i] = wholeX[tempSequence[i]];
			trainingY[i] = wholeY[tempSequence[i]];
		} // Of for i

		// Step 3. Copy the testing set.
		int tempTestingSize = wholeY.length - tempTrainingSize;
		testingX = new double[tempTestingSize][];
		testingY = new double[tempTestingSize][];
		for (int i = 0; i < tempTestingSize; i++) {
			testingX[i] = wholeX[tempSequence[tempTrainingSize + i]];
			testingY[i] = wholeY[tempSequence[tempTrainingSize + i]];
		} // Of for i

		//System.out.println("The training X is: " + Arrays.deepToString(trainingX));
		//System.out.println("The training Y is: " + Arrays.deepToString(trainingY));
		//System.out.println("The testing X is: " + Arrays.deepToString(testingX));
		//System.out.println("The testing Y is: " + Arrays.deepToString(testingY));
	}// Of randomizeTrainingTesting

	/**
	 ****************** 
	 * Train the self-paced-regressor.
	 ****************** 
	 */
	public double[] train() {
		// Step 1. Build the original hyperplane.
		System.out.println("Training ... the training set has " + trainingX.length + " instances.");
		beforeinstance=trainingX.length;
		weights = train(trainingX, trainingY);
		//System.out.println("All data, the weights are: " + Arrays.toString(weights));
		double tempMAE = computeTestingMae();
		System.out.println("The MAE with all training data is: " + tempMAE);
		tempValue=tempMAE;
		// Step 2. Increase distance gradually.
		double tempDistanceThreshold = distanceThresholdInitial;
		double tempNumNeighbors = 0;
		for (int i = 0; i < 10; i++) {
			// Step 2.1 Select a subset.
			int[] tempIndices = select(weights, tempDistanceThreshold);
			tempNumNeighbors = tempIndices.length;

			double[][] tempX = new double[tempIndices.length][];
			double[][] tempY = new double[tempIndices.length][];
			//System.out.println("TrainingY: " + Arrays.deepToString(trainingY));
			// Copy data
			for (int j = 0; j < tempX.length; j++) {
				// System.out.print(", " + tempIndices[j] + ": " +
				// trainingY[tempIndices[j]][0]);
				tempX[j] = trainingX[tempIndices[j]];
				tempY[j] = trainingY[tempIndices[j]];
			} // Of for j

			// Step 2.2 Update the weights
			weights = train(tempX, tempY);

			// Step 2.3 Not all data are useful
			if (tempNumNeighbors > trainingX.length * 0.8) {
				// Enough training data are used.
				break;
			} // Of if

			// Step 2.4 Increase the distance threshold
			tempDistanceThreshold += distanceThresholdIncrement;
		} // Of for i

		System.out.println("Finally, the threshold is " + tempDistanceThreshold
				+ " with " + tempNumNeighbors + " neighbors.");
		tempThreshold=tempDistanceThreshold;
		afterinstance=tempNumNeighbors;
		//System.out.println("The weights are: " + Arrays.toString(weights));
		return weights;
	}// Of train

	/**
	 ****************** 
	 * Train with the given data matrices.
	 * 
	 * @param paraX
	 *            The input data.
	 * @param paraY
	 *            The output data.
	 * @return The weight vector.
	 ****************** 
	 */
	public double[] train(double[][] paraX, double[][] paraY) {
		// System.out.println("paraX: " + Arrays.deepToString(paraX));
		// System.out.println("paraY: " + Arrays.deepToString(paraY));

		// Step 1. Construct a matrix object.
		Matrix tempX = new Matrix(paraX);
		Matrix tempY = new Matrix(paraY);

		// Step 2. Regression directly.
		// (X T X)-1 XT y
		Matrix tempMatrix = (tempX.transpose().times(tempX)).inverse().times(tempX.transpose())
				.times(tempY);
		double[] resultWeights = tempMatrix.transpose().getArray()[0];

		// System.out.println("The new weights are: " +
		// Arrays.toString(resultWeights));

		return resultWeights;
	}// Of train

	/**
	 ****************** 
	 * Select data close to the hyperplane.
	 * 
	 * @param paraWeights
	 *            The hyperplane.
	 * @param paraDistance
	 *            The distance.
	 * @return The data indices.
	 ****************** 
	 */
	public int[] select(double[] paraWeights, double paraDistance) {
		int[] tempSelectionArray = new int[trainingX.length];
		double tempDistance = 0;
		int tempNumSelection = 0;
		for (int i = 0; i < trainingX.length; i++) {
			double tempPrediction = 0;
			for (int j = 0; j < trainingX[0].length; j++) {
				tempPrediction += paraWeights[j] * trainingX[i][j];
			} // Of for j
			tempDistance = Math.abs(tempPrediction - trainingY[i][0]);

			if (tempDistance < paraDistance) {
				tempSelectionArray[tempNumSelection] = i;
				tempNumSelection++;
			} // Of if
		} // Of for i

		System.out.println("" + tempNumSelection + " instances are close to the hyperplane.");

		// Compress
		int[] resultSelections = new int[tempNumSelection];
		for (int i = 0; i < resultSelections.length; i++) {
			resultSelections[i] = tempSelectionArray[i];
		} // Of for i

		return resultSelections;
	}// Of select

	/**
	 ****************** 
	 * Regress for an instance represented by an array.
	 * 
	 * @param paraInstance
	 *            The given instance.
	 * @return The predicted label (numerical).
	 ****************** 
	 */
	public double regress(double[] paraInputArray) {
		double result = 0;
		for (int i = 0; i < paraInputArray.length; i++) {
			result += weights[i] * paraInputArray[i];
		} // Of for i

		return result;
	}// Of regress

	/**
	 ****************** 
	 * Compute the mean absolute error on the testing set.
	 * 
	 * @param paraInstances
	 *            The testing set.
	 * @return The mean absolute error.
	 ****************** 
	 */
	public double computeTestingMae() {
		double tempErrorSum = 0;
		double tempError=0;
		for (int i = 0; i < testingX.length; i++) {
			double tempPredict = 0;
			for (int j = 0; j < testingX[0].length; j++) {
				tempPredict += testingX[i][j] * weights[j];
			}//Of for j
			
			//tempErrorSum += Math.abs(tempPredict - testingY[i][0]);
			tempError= Math.abs(tempPredict - testingY[i][0]);
			tempErrorSum +=tempError*tempError;
		} // Of for i

		return tempErrorSum / testingX.length;
	}// Of computeTestingMae
	
	/**
	 ****************** 
	 * For integration test.
	 * 
	 * @param args
	 *            Not provided.
	 * @throws IOException 
	 ****************** 
	 */
	public static void main(String args[]) throws IOException {
		System.out.println("Starting self-paced regression ...");
		//SelfPacedRegressor tempSelfPacedRegressor = new SelfPacedRegressor("src/data/iris.arff");
		//SelfPacedRegressor tempSelfPacedRegressor = new SelfPacedRegressor("src/data/meta-test/500s.arff");
		String[]label_list= {"bank8fm.arff","Concrete_Data.arff","dataset_2211_puma8NH.arff","dataset_2213_cpu_small.arff",
		                    "delta_elevators.arff","EGSSD.arff","Folds5x2_pp.arff","fried.arff","house_16H.arff","kin8nm.arff","wine_quality.arff"};
		
		String datapath= "src/data/";
		String tempdataPath="";
		File file = new File("SPLresult-MSE.txt");
		FileWriter out = new FileWriter(file);
		for (int i = 0; i < label_list.length; i++) {
			tempdataPath=datapath+label_list[i];
			//System.out.println(tempdataPath);
			SelfPacedRegressor tempSelfPacedRegressor = new SelfPacedRegressor(tempdataPath);
			double tempLog=0;
			for (int j = 0; j < 100; j++) {
				tempSelfPacedRegressor.randomizeTrainingTesting(0.1);
				double[] tempWeights = tempSelfPacedRegressor.train();
				double tempMAE = tempSelfPacedRegressor.computeTestingMae();
				tempLog+=tempMAE;
			}
			out.write(tempLog/100+"\r\n");
		}
				
		// SelfPacedRegressor tempSelfPacedRegressor = new
		// SelfPacedRegressor("src/data/iris.arff", 200);
		
		/**double tempRise=0;
		for (int i = 0; i < 100; i++) {			
			tempValue=0;
			tempThreshold=0;
			beforeinstance=0;
			afterinstance=0;
			
			out.write(tempValue+"\t");
			out.write(tempMAE+"\t");
			out.write(tempThreshold+"\t");
			out.write(beforeinstance+"\t");
			out.write(afterinstance+"\t");
			out.write(((tempValue-tempMAE)/tempValue)*100+"\r\n" );
			tempRise+=((tempValue-tempMAE)/tempValue);
			System.out.println("The MAE with selected data is: " + tempMAE);
		}
		*/
		//out.write( tempRise+"\t");
		out.close();	
		
		// tempSelfPacedRegressor.select(tempWeights, 3.0);

		// System.out.println("The training mae is: " +
		// tempSelfPacedRegressor.computeMae());
	}// Of main

}// Of class SelfPacedRegressor