package algorithm;

import java.io.BufferedReader;
/**
 * The learning method of Spor. 
 * It trains the training set after the Spor process and compared with traditional co-training method.
 * 
 * @author Yu Li<br>
 *         Email:1132559357@qq.com<br>
 *         Date Created£ºAugust 5, 2020 <br>
 *         Last Modifide: August 8, 2020 <br>
 * 
 * @version 1.0
 */
import java.io.FileReader;
import java.text.DecimalFormat;
import common.DistanceMeasure;
import common.SimpleTools;
import cotrainer.Cotrainer;
import weka.core.Instance;
import weka.core.Instances;

/**
 * The learning method of SPOR. It trains given data based self-pace co-training regrime.
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * @author Yu Li<br>
 *         Email:1132559357@qq.com<br>
 *         Date Created£ºAugust 10, 2020 <br>
 *         Last Modifide: August 20, 2020 <br>
 * 
 * @version 1.0
 */
public class Learner {
	/**
	 * The training set.
	 */
	Instances trainingSet;

	/**
	 * The unlabeled set.
	 */
	Instances unlabeledSet;

	/**
	 * The testing set.
	 */
	Instances testingSet;

	/**
	 * The data.
	 */
	Instances data;

	/**
	 * The size of the training set.
	 */
	int trainingSetSize;

	/**
	 * The size of the unlabele'd set.
	 */
	int unlabelSetSize;

	/**
	 * The size of the testing set.
	 */
	int testingSetSize;

	/**
	 * The kValue of the first cotrainer.
	 */
	int firstKvalue;

	/**
	 * The kValue of the second cotrainer.
	 */
	int secondKvalue;

	/**
	 * The pool size of unlabeled set.
	 */
	static int poolSize = 100;

	/**
	 * The training iteration.
	 */
	int iterations;

	/**
	 * The self-pace-lambda to select unlabeled instance.
	 */
	double Lambda = 1;

	/**
	 * The self-pace-gamma to fix unlabeled instance.
	 */
	double gamma = 0.01;

	/**
	 * The adoptive array based the value of gamma to fix the instance selection.
	 */
	double[] gammaArray = new double[poolSize];

	/**
	 * The step size of lambda.
	 */
	double stepSize = 0;

	/**
	 * The drop rate of cortrainer.
	 */
	public double errorDrop = 0;

	/**
	 * The number of instances added by first cortrainer.
	 */
	public int firstCortaininerAddInstances = 0;

	/**
	 * The number of instances added by second cortrainer.
	 */
	public int secondCortaininerAddInstances = 0;

	/**
	 * The maximum drop rate of cortrainer.
	 */
	public double maxErrorDrop = 0;

	/**
	 * The minimum drop rate of cortrainer.
	 */
	public double minErrorDrop = 0;

	/**
	 * The mean squared error of cortrainer before co-training process.
	 */
	public double beforeMse = 0;

	/**
	 * The mean squared error of cortrainer after co-training process.
	 */
	public double afterMse = 0;

	/**
	 * The distance measure (such as Manhattan, Euclidean distance) of the first cotrainer.
	 */
	DistanceMeasure firstDistanceMeasure;

	/**
	 * The distance measure (such as Manhattan, Euclidean distance) of the second cotrainer.
	 */
	DistanceMeasure secondDistanceMeasure;

	/**
	 * The first cotrainer.
	 */
	public Cotrainer firstCotrainer;

	/**
	 * The second cotrainer.
	 */
	public Cotrainer secondCotrainer;

	/**
	 ********************
	 * Initialize log parameters.
	 ********************
	 */
	public void initializeParameters() {
		errorDrop = 0;
		maxErrorDrop = 0;
		minErrorDrop = 0;
		firstCortaininerAddInstances = 0;
		secondCortaininerAddInstances = 0;
		beforeMse = 0;
		afterMse = 0;
	}// Of setParameters

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraData             The given data.
	 * @param paraDistanceMeasure1 The given distance measure of the first cotraniner.
	 * @param paraDistanceMeasure2 The given distance measure of the second cotraniner.
	 * @param paraKvalue1          The given kValue of the first cotraniner.
	 * @param parakvalue2          The given kValue of the second cotraniner.
	 * @param paraPoolsize         The given pool size of the unlableledSet.
	 * @param paraLabeledrate      The proportion of the training data.
	 * @param paraTestingrate      The proportion of the testing data.
	 * @param paraiterations       The training iterations.
	 * @param paraLambda           The self-pace lambda.
	 * @param paraStepSize         The step size to control lambda.
	 * @param paraGamma			   The gamma value to control instance selection.
	 ********************
	 */
	public Learner(Instances paraData, int paraDistanceMeasure1, int paraDistanceMeasure2, int paraKvalue1,
			int paraPoolsize, int parakvalue2, double paraLabeledrate, double paraTestingrate, int paraiterations,
			double paraLambda, double paraStepSize, double paraGamma) {

		// Step 1 Set the parameters of constructor.
		data = paraData;
		Lambda = paraLambda;
		gamma = paraGamma;
		stepSize = paraStepSize;
		firstKvalue = paraKvalue1;
		secondKvalue = parakvalue2;
		poolSize = paraPoolsize;
		iterations = paraiterations;
		trainingSetSize = (int) (data.numInstances() * paraLabeledrate);
		testingSetSize = (int) (data.numInstances() * paraTestingrate);
		unlabelSetSize = data.numInstances() - trainingSetSize - testingSetSize;
		trainingSet = new Instances(data, 0);
		testingSet = new Instances(data, 0);
		unlabeledSet = new Instances(data, 0);
		gammaArray = gammaArray(paraGamma);
		Instance tempInstance = null;

		// Step 2 Split the data set.
		for (int i = 0; i < trainingSetSize; i++) {
			tempInstance = data.instance(i);
			trainingSet.add(tempInstance);
		} // Of for i
		for (int i = trainingSetSize; i < trainingSetSize + testingSetSize; i++) {
			tempInstance = data.instance(i);
			unlabeledSet.add(tempInstance);
		} // Of for i
		for (int i = trainingSetSize + testingSetSize; i < data.numInstances(); i++) {
			tempInstance = data.instance(i);
			testingSet.add(tempInstance);
		} // Of for i

		// Step 3 Build the cotrainer.
		firstCotrainer = new Cotrainer(paraKvalue1, trainingSet, unlabeledSet, testingSet, paraPoolsize,
				paraDistanceMeasure1, Lambda);
		secondCotrainer = new Cotrainer(parakvalue2, trainingSet, unlabeledSet, testingSet, paraPoolsize,
				paraDistanceMeasure2, Lambda);
	}// Of learner

	/**
	 ********************
	 * Cotraining process.
	 * 
	 * Two Cotrainer select the most confidence instance for partner from unlabeledSet.
	 ********************
	 */
	public void Cotraining() {
		// Step 1 Initialize the given instance
		Instance firstInstance;
		Instance secondInstance;

		// Step 2 Find the instance for other regressor and add it into training set.
		for (int i = 0; i < iterations; i++) {
			int cotrainingFlag = 2;
			firstInstance = firstCotrainer.selectCriticalInstance();
			if (firstInstance == null) {
				cotrainingFlag -= 1;
			} else {
				secondCotrainer.updateTrainingSet(firstInstance);
				secondCortaininerAddInstances++;
			} // Of if
			secondInstance = secondCotrainer.selectCriticalInstance();

			if (secondInstance == null) {
				cotrainingFlag -= 1;
			} else {
				firstCotrainer.updateTrainingSet(secondInstance);
				firstCortaininerAddInstances++;
			} // Of if
			if (cotrainingFlag == 0) {
				break;
			} // Of if
		} // Of for i
	}// Of Cotraining

	/**
	 ********************
	 * Cotraining process.
	 * 
	 * Two Cotrainer select the most confidence instance for partner from unlabeledSet.
	 ********************
	 */
	public void Cotraininges() {
		// Step 1. Initialize the given instances
		Instances firstAddedInstances;
		Instances secondAddedInstances;

		// Step 2. Find the confidence instances for other regressor and add them into training set.
		for (int i = 0; i < iterations; i++) {
			int cotrainingFlag = 2;
			firstAddedInstances = firstCotrainer.selectCriticalInstances();
			if (firstAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				secondCotrainer.updateTrainingSet(firstAddedInstances);
				secondCortaininerAddInstances += firstAddedInstances.numInstances();
			} // Of if

			secondAddedInstances = secondCotrainer.selectCriticalInstances();
			if (secondAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				firstCotrainer.updateTrainingSet(secondAddedInstances);
				firstCortaininerAddInstances += secondAddedInstances.numInstances();
			} // Of if
			if (cotrainingFlag == 0) {
				break;
			} // Of if
		}// Of for i
	}// Of Cotraininges

	/**
	 ********************
	 * Self-pace Cotraining process.
	 * 
	 * Two Cotrainer select the most confidence instance for partner from unlabeledSet.
	 ********************
	 */
	public void splCotraininges() {
		// Step 1 Initialize the given instances
		Instances firstAddedInstances;
		Instances secondAddedInstances;

		// Step 2 Using self-pace Lambda to find the confidence instances for other regressor and add them
		// into training set.
		double tempLambda = Lambda;
		for (int i = 0; i < iterations; i++) {
			tempLambda = Lambda - i * stepSize;
			int cotrainingFlag = 2;
			firstCotrainer.setLambda(tempLambda);
			firstAddedInstances = firstCotrainer.selectCriticalInstances();
			if (firstAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				secondCotrainer.updateTrainingSet(firstAddedInstances);
				secondCortaininerAddInstances += firstAddedInstances.numInstances();

			} // Of if
			secondCotrainer.setLambda(tempLambda);
			secondAddedInstances = secondCotrainer.selectCriticalInstances();
			if (secondAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				firstCotrainer.updateTrainingSet(secondAddedInstances);
				firstCortaininerAddInstances += secondAddedInstances.numInstances();
			} // Of if
			if (cotrainingFlag == 0) {
				break;
			} // Of if
		}// Of for i
	}// Of splCotraininges

	/**
	 ********************
	 * Self-pace Cotraining process.
	 * 
	 * Two Cotrainer select the most confidence instance for partner from unlabeledSet.
	 ********************
	 */
	public void spmcoCotraininges() {
		// Step 1 Initialize the given instances
		Instances firstAddedInstances;
		Instances secondAddedInstances;

		// Step 2 Using self-pace lambda to find the confidence instances for other regressor and add them
		// into training set.
		double tempLambda = Lambda;
		for (int i = 0; i < iterations; i++) {
			tempLambda = Lambda - i * stepSize;
			int cotrainingFlag = 2;
			firstCotrainer.setLambda(tempLambda);
			firstAddedInstances = firstCotrainer.selectCriticalInstances(secondCotrainer.getTrainingSet(), gammaArray,
					secondKvalue);
			if (firstAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				secondCotrainer.updateTrainingSet(firstAddedInstances);
				secondCortaininerAddInstances += firstAddedInstances.numInstances();
			} // Of if
			secondCotrainer.setLambda(tempLambda);
			secondAddedInstances = secondCotrainer.selectCriticalInstances(firstCotrainer.getTrainingSet(), gammaArray,
					firstKvalue);
			if (secondAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				firstCotrainer.updateTrainingSet(secondAddedInstances);
				firstCortaininerAddInstances += secondAddedInstances.numInstances();
			} // Of if
			if (cotrainingFlag == 0) {
				break;
			} // Of if
		}// Of for i
	}// Of spmcoCotraininges

	/**
	 ********************
	 * Self-pace Cotraining process.
	 * 
	 * Two Cotrainer select the most confidence instance for partner from unlabeledSet with stable
	 * Lambda.
	 ********************
	 */
	public void stableCotraininges() {
		// Step 1 Initialize the given instances
		Instances firstAddedInstances;
		Instances secondAddedInstances;

		// Step 2 Using self-pace Lambda to find the confidence instances for other regressor and add them
		// into training set.

		for (int i = 0; i < iterations; i++) {
			int cotrainingFlag = 2;
			firstAddedInstances = firstCotrainer.selectCriticalInstances(secondCotrainer.getTrainingSet(), gammaArray,
					secondKvalue);
			if (firstAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				secondCotrainer.updateTrainingSet(firstAddedInstances);
				secondCortaininerAddInstances += firstAddedInstances.numInstances();
			} // Of if
			secondAddedInstances = secondCotrainer.selectCriticalInstances(firstCotrainer.getTrainingSet(), gammaArray,
					firstKvalue);
			if (secondAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				firstCotrainer.updateTrainingSet(secondAddedInstances);
				firstCortaininerAddInstances += secondAddedInstances.numInstances();
			} // Of if
			if (cotrainingFlag == 0) {
				break;
			} // Of if
		}// Of for i
	}// Of stableCotraininges

	/**
	 ********************
	 * The gamma term to correct the instance selection strategy.
	 * 
	 * @param paraGamma. The maximum value of gamma.
	 * @return resultGamma. The adoptive value of gamma.
	 ********************
	 */
	public static double[] gammaArray(double paraGamma) {
		double tempW = 0;
		tempW = paraGamma * 4 / (poolSize * poolSize);
		double[] resultGamma = new double[poolSize];
		for (int i = 0; i < resultGamma.length; i++) {
			if (i < 0.5 * poolSize) {
				resultGamma[i] = -1 * tempW * (i - poolSize / 2) * (i - poolSize / 2);
			} else {
				resultGamma[i] = tempW * (i - poolSize / 2) * (i - poolSize / 2);
			}// Of if
		}// Of for
		return resultGamma;
	}// Of gammaArray

	/**
	 ********************
	 * Learn process. Test the effect of Cotraining process.
	 * 
	 * @param paraCotrainer1 The first Cotrainer.
	 * @param paraCotrainer2 The second Cotrainer.
	 * @return The running message.
	 ********************
	 */
	public String Learn(Cotrainer paraCotrainer1, Cotrainer paraCotrainer2) {
		// Step 1 Initialize the tool parameters and arrays.
		DecimalFormat decimalFormat = new DecimalFormat("0.000000000");
		double tempMse1 = 0;
		double tempMse2 = 0;
		Instance tempInstance = null;
		String resultMessage = "";

		// Step 2 Test the model in testing set.
		for (int i = 0; i < testingSet.numInstances(); i++) {

			// Step 2.1 Compute the prediction of testing instances and mean squared error.
			tempInstance = testingSet.instance(i);
			double tempValue1 = (paraCotrainer1.predict(tempInstance) + paraCotrainer2.predict(tempInstance)) / 2;
			double tempValue2 = (paraCotrainer1.predict(trainingSet, tempInstance)
					+ paraCotrainer2.predict(trainingSet, tempInstance)) / 2;
			tempMse1 += (tempValue1 - tempInstance.classValue()) * (tempValue1 - tempInstance.classValue());
			tempMse2 += (tempValue2 - tempInstance.classValue()) * (tempValue2 - tempInstance.classValue());
		} // Of for i
		tempMse1 /= testingSet.numInstances();
		tempMse2 /= testingSet.numInstances();

		// Step 2.2 Log the performance of the model.
		beforeMse += tempMse2;
		afterMse += tempMse1;
		double tempMseDrop = 0;
		tempMseDrop = tempMse2 - tempMse1;
		errorDrop += (tempMseDrop / tempMse2);
		if (maxErrorDrop < (tempMseDrop)) {
			maxErrorDrop = tempMseDrop;
		} // Of if
		if (minErrorDrop > (tempMseDrop)) {
			minErrorDrop = tempMseDrop;
		} // Of if
		// Step 2.3 Print the test informations.
		resultMessage += "Mean-square error:" + decimalFormat.format(tempMse2) + "(" + trainingSet.numInstances() + "/"
				+ trainingSet.numInstances() + ")" + "->";
		resultMessage += "" + decimalFormat.format(tempMse1) + "(" + firstCotrainer.getTrainingSetSize() + "/"
				+ secondCotrainer.getTrainingSetSize() + ")" + "      ";
		resultMessage += "The error drop:" + decimalFormat.format(tempMseDrop);
		resultMessage += "      Drop Rate:" + decimalFormat.format((tempMseDrop / tempMse2) * 100) + "%" + "\r\n";
		return resultMessage;
	}// Of learn

	/**
	 ************************* 
	 * Test this class.
	 *
	 * @param args The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		String tempString = "";
		try {
			BufferedReader r = new BufferedReader(new FileReader("src/data/kin8nm.arff"));
			Instances tempdata = new Instances(r);
			r.close();
			tempdata.setClassIndex(tempdata.numAttributes() - 1);
			SimpleTools.normalizeDecisionSystem(tempdata);
			double[] resultGamma = gammaArray(0.3);
			System.out.print(resultGamma);
			for (int i = 0; i < 10; i++) {
				SimpleTools.disorderData(tempdata);
				Learner tempLearner = new Learner(tempdata, DistanceMeasure.EUCLIDEAN, DistanceMeasure.EUCLIDEAN, 3, 50,
						3, 0.1, 0.5, 20, 0.9, 0.05, 0.01);
				tempLearner.Cotraining();
				tempString += tempLearner.Learn(tempLearner.firstCotrainer, tempLearner.secondCotrainer);
			} // Of for i
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try
		System.out.println(tempString);
	}// Of main
}// Of class learner
