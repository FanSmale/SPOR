package algorithm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.PrimitiveIterator.OfDouble;

import common.DistanceMeasure;
import common.SimpleTools;
import cotrainer.Cotrainer;
import regressor.KnnRegressor;
import weka.classifiers.pmml.consumer.Regression;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.FirstOrder;

import java.awt.Label;
import java.io.*;
import java.util.*;
import java.text.*;

import weka.core.*;

public class Learner {
	/**
	 * The trainingSet.
	 */
	Instances trainingSet;
	/**
	 * The unlabeledSet.
	 */
	Instances unlabeledSet;
	/**
	 * The testingSet.
	 */
	Instances testingSet;
	/**
	 * The data.
	 */
	Instances data;
	/**
	 * The size of the trainingSet.
	 */
	int trainingSetsize;
	/**
	 * The size of the unlabelSetsize.
	 */
	int unlabelSetsize;
	/**
	 * The size of the testingSetsize.
	 */
	int testingSetsize;
	/**
	 * The kvalue of the first cotrainer.
	 */
	int firstKvalue;
	/**
	 * The kvalue of the second cotrainer.
	 */
	int secondKvalue;
	/**
	 * The poolSize.
	 */
	int poolSize;
	/**
	 * The training iterations.
	 */
	int iterations;
	/**
	 * The denoise effective times.
	 */
	double Threshold;
	public static int win = 0;
	/**
	 * The denoise does not effective times.
	 */
	public static int lose = 0;
	/**
	 * The DistanceMeasure of the first cotrainer.
	 */
	DistanceMeasure firstDistanceMeasure;
	/**
	 * The DistanceMeasure of the second cotrainer.
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
	 * The constructor.
	 * 
	 * @param paraData             The given data.
	 * @param paraDistanceMeasure1 The given distance measure of the first cotraniner.
	 * @param paraDistanceMeasure2 The given distance measure of the second cotraniner.
	 * @param paraKvalue1          The given kValue of the first cotraniner.
	 * @param paraKvalue2          The given Kvalue of the second cotraniner.
	 * @param paraPoolsize         The given poolsize of the unlableledSet.
	 * @param paraLabeledrate      The proportion of the training data.
	 * @param paraTestingrate      The proportion of the testing data.
	 * @param paraIterations       The training iterations.
	 ********************
	 */
	public Learner(Instances paraData, int paraDistanceMeasure1, int paraDistanceMeasure2, int paraKvalue1,
			int paraPoolsize, int paraKvalue2, double paraLabeledrate, double paraTestingrate, int paraIterations,double paraThreshold) {
		data = paraData;
		Threshold =paraThreshold;
		firstKvalue = paraKvalue1;
		secondKvalue = paraKvalue2;
		poolSize = paraPoolsize;
		iterations = paraIterations;
		trainingSetsize = (int) (data.numInstances() * paraLabeledrate);
		testingSetsize = (int) (data.numInstances() * paraTestingrate);
		unlabelSetsize = data.numInstances() - trainingSetsize - testingSetsize;
		trainingSet = new Instances(data, 0);
		testingSet = new Instances(data, 0);
		unlabeledSet = new Instances(data, 0);
		Instance tempInstance = null;
		for (int i = 0; i < trainingSetsize; i++) {
			tempInstance = data.instance(i);
			trainingSet.add(tempInstance);
		} // of for i
		for (int i = trainingSetsize; i < trainingSetsize + testingSetsize; i++) {
			tempInstance = data.instance(i);
			unlabeledSet.add(tempInstance);
		} // of for i
		for (int i = trainingSetsize + testingSetsize; i < data.numInstances(); i++) {
			tempInstance = data.instance(i);
			testingSet.add(tempInstance);
		} // of for i
		firstCotrainer = new Cotrainer(paraKvalue1, trainingSet, unlabeledSet, testingSet, paraPoolsize,
				paraDistanceMeasure1,Threshold);
		secondCotrainer = new Cotrainer(paraKvalue2, trainingSet, unlabeledSet, testingSet, paraPoolsize,
				paraDistanceMeasure2,Threshold);
	}// of learner

	/**
	 ********************
	 * Cotraining process.Two Cotrainer select the most confidengce instance for partner from
	 * unlabeledSet.
	 ********************
	 */
	public void cotraining() {
		Instance firstInstance;
		Instance secondInstance;

		for (int i = 0; i < iterations; i++) {
			int cotrainingFlag = 2;
			firstInstance = firstCotrainer.selectCriticalInstance();
			if (firstInstance == null) {
				cotrainingFlag -= 1;
			} else {
				secondCotrainer.updateTrainingSet(firstInstance);
				SimpleTools.NumInstances2added++;
			
			} // of if
			secondInstance = secondCotrainer.selectCriticalInstance();

			if (secondInstance == null) {
				cotrainingFlag -= 1;
			} else {
				firstCotrainer.updateTrainingSet(secondInstance);
				SimpleTools.NumInstances1added++;
			
			} // of if

			if (cotrainingFlag == 0) {
				break;
			} // of if
		} // of for i
	}// of cotraining

	public void Cotraininges() {
		Instances firstAddedInstances;
		Instances secondAddedInstances;
		double stepsize=0.005;
		double tempThreshold=0;
		for (int i = 0; i < iterations; i++) {
			tempThreshold=Threshold-i*stepsize;
			int cotrainingFlag = 2;
			//firstCotrainer.setThreshold(tempThreshold);
			firstAddedInstances = firstCotrainer.selectCriticalInstances();
			if (firstAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				secondCotrainer.updateTrainingSet(firstAddedInstances);
				SimpleTools.NumInstances2added+=firstAddedInstances.numInstances();
				// System.out.println(SimpleTools.NumInstances2added);
			} // of if	
			//secondCotrainer.setThreshold(tempThreshold);
			secondAddedInstances = secondCotrainer.selectCriticalInstances();
			if (secondAddedInstances == null) {
				cotrainingFlag -= 1;
			} else {
				firstCotrainer.updateTrainingSet(secondAddedInstances);
				SimpleTools.NumInstances1added+=secondAddedInstances.numInstances();
				// System.out.println(SimpleTools.NumInstances2added);
			} // of if
			if (cotrainingFlag == 0) {
				break;
			} // of if
		}	
	}
	

	/**
	 ********************
	 * Learn. Test the effect of Cotraining process.
	 * 
	 * @param paraCotrainer1 The first Cotrainer.
	 * @param paraCotrainer2 The second Cotrainer.
	 * @return The running message.
	 ********************
	 */
	public String Learn(Cotrainer paraCotrainer1, Cotrainer paraCotrainer2) {
		DecimalFormat df = new DecimalFormat("0.000000000");
		double tempMse1 = 0;
		double tempMse2 = 0;
		double tempPreMse = 0;
		double tempDeNoiseMse = 0;
		Instance tempInstance = null;
		String resultMessage = "";
		//paraCotrainer1.crossValid();
		//paraCotrainer2.crossValid();
		for (int i = 0; i < testingSet.numInstances(); i++) {
			tempInstance = testingSet.instance(i);
			double tempValue1 = (paraCotrainer1.predict(tempInstance) + paraCotrainer2.predict(tempInstance)) / 2;
			//double testpredict1 = paraCotrainer1.predict(tempInstance);
			//double testpredict2 = paraCotrainer2.predict(tempInstance);
			double tempValue2 = (paraCotrainer1.predict(trainingSet, tempInstance)
					+ paraCotrainer2.predict(trainingSet, tempInstance)) / 2;
			//double testpredict3 = paraCotrainer1.predict(trainingSet, tempInstance);
			//double testpredict4 = paraCotrainer2.predict(trainingSet, tempInstance);
			double tempDeNoiseValue = (paraCotrainer1.deNoisePredict(tempInstance)
					+ paraCotrainer2.deNoisePredict(tempInstance)) / 2;
			double tempPrevalue = (paraCotrainer1.preDeNoisePredict(tempInstance)
					+ paraCotrainer2.preDeNoisePredict(tempInstance)) / 2;
			tempPreMse+=(tempPrevalue - tempInstance.classValue())
					* (tempPrevalue - tempInstance.classValue());
			tempMse1 += (tempValue1 - tempInstance.classValue()) * (tempValue1 - tempInstance.classValue());
			tempMse2 += (tempValue2 - tempInstance.classValue()) * (tempValue2 - tempInstance.classValue());
			tempDeNoiseMse += (tempDeNoiseValue - tempInstance.classValue())
					* (tempDeNoiseValue - tempInstance.classValue());
		} // of for i
		tempMse1 /= testingSet.numInstances();
		tempMse2 /= testingSet.numInstances();
		tempDeNoiseMse /= testingSet.numInstances();
		tempPreMse/=testingSet.numInstances();
		if (tempDeNoiseMse <= tempMse1) {
			SimpleTools.win++;
		} else {
			SimpleTools.lose++;
		}
		double tempMseDrop = 0;
		tempMseDrop = tempMse2 - tempMse1;
		SimpleTools.errorDrop += (tempMseDrop / tempMse2);
		SimpleTools.lastDrop += (tempMse2 - tempDeNoiseMse) / tempMse2;
		SimpleTools.preDrop+=tempMse1;
		if (SimpleTools.maxErrorDrop < (tempMseDrop)) {
			SimpleTools.maxErrorDrop = tempMseDrop;
		} // of if
		if (SimpleTools.minErrorDrop > (tempMseDrop)) {
			SimpleTools.minErrorDrop = tempMseDrop;
		} // of if
		resultMessage += "Mean-square error:" + df.format(tempMse2) +"("+trainingSet.numInstances()+"/"+trainingSet.numInstances()+")"+ "->";
		resultMessage += "" + df.format(tempMse1) +"("+firstCotrainer.trainingSetSize()+"/"+secondCotrainer.trainingSetSize()+")"+"      ";
		resultMessage += "The error drop:" + df.format(tempMseDrop);
		resultMessage += "      Drop Rate:" + df.format((tempMseDrop / tempMse2) * 100) + "%" + "\r\n";
		resultMessage += "The denoise Mse:" + df.format(tempMse1) +"("+firstCotrainer.trainingSetSize()+"/"+secondCotrainer.trainingSetSize()+")"+ "->" + df.format(tempDeNoiseMse) +"("+firstCotrainer.debugtrainingSetSize()+"/"+secondCotrainer.debugtrainingSetSize()+")"+ "      ";
		resultMessage += "The denoise Mse drop rate:" + df.format(((tempMse1 - tempDeNoiseMse) / tempMse1) * 100) + "%"
				+ "\r\n";
		resultMessage += "The last drop:" + df.format(((tempMse2 - tempDeNoiseMse) / tempMse2) * 100) + "%" + "\r\n";
		return resultMessage;
	}// of learn

	/**
	 ************************* 
	 * Test this class.
	 *
	 * @param args The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		String tempFilename = "/Coregtest/src/data/kin8nm.arff";
		String tempString = "";
		try {
			BufferedReader r = new BufferedReader(new FileReader("src/data/kin8nm.arff"));
			Instances tempdata = new Instances(r);
			r.close();
			tempdata.setClassIndex(tempdata.numAttributes() - 1);
			SimpleTools.normalizeDecisionSystem(tempdata);
			for (int i = 0; i < 10; i++) {
				SimpleTools.disorderData(tempdata);
				Learner tempLearner = new Learner(tempdata, DistanceMeasure.EUCLIDEAN, DistanceMeasure.EUCLIDEAN, 3, 50,
						3, 0.1, 0.5, 20,0.9);
				tempLearner.cotraining();
				tempLearner.firstCotrainer.denoise();
				tempLearner.secondCotrainer.denoise();
				tempString += tempLearner.Learn(tempLearner.firstCotrainer, tempLearner.secondCotrainer);
			} // of for i
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
		System.out.println(tempString);
	}// of main
}// of class learner
