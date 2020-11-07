package gui;

import java.awt.*;
import java.awt.event.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import javax.swing.JComboBox;

import algorithm.*;
import common.DistanceMeasure;
import common.SimpleTool;
import common.SimpleTools;
import gui.guicommon.*;
import gui.guidialog.common.ErrorDialog;
import gui.guidialog.common.HelpDialog;
import gui.others.*;
import weka.core.Instances;


/**
 * Self-pace co-training using attribute reduct pairs. 
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
public class SporGUI implements ActionListener {
	/**
	 * Select the arff file.
	 */
	private FilenameField arffFilenameField;

	/**
	 * The ratio of labeled data.
	 */
	private DoubleField labelFractionField;

	/**
	 * The ratio of test data.
	 */
	private DoubleField testFractionField;
	/**
	 * The ratio of unlabeled data.
	 */
	private IntegerField trainingIterations;

	/**
	 * Co-training measures: Coreg, Spor, Plain
	 */
	private JComboBox<String> algorithmJComboBox;

	/**
	 * Distance measures: Euclidean, Manhattan, Mahalanobis
	 */
	private JComboBox<String> distanceJComboBox1;

	/**
	 * Distance measures: Euclidean, Manhattan, Mahalanobis
	 */
	private JComboBox<String> distanceJComboBox2;

	/**
	 * Normalize or not.
	 */
	private Checkbox normalizeCheckbox;

	/**
	 * Checkbox for self-pace term.
	 */
	private Checkbox lamdaCheckbox;

	/**
	 * Checkbox for gamma term.
	 */
	private Checkbox gammaCheckbox;

	/**
	 * Checkbox for step size.
	 */
	private Checkbox stepSizecheckbox;

	/**
	 * Disorder or not.
	 */
	private Checkbox disorderCheckbox;

	/**
	 * The step size to control the value of self-pace {@link Terminable#}
	 */
	private DoubleField stepSizeField;

	/**
	 * The k value for first Cotrainer.
	 */
	private IntegerField kValueIntegerField1;
	/**
	 * The k value for second Cotrainer.
	 */
	private IntegerField kValueIntegerField2;

	/**
	 * The self-pace term to control the instance selection.
	 */
	private DoubleField lamdaField;

	/**
	 * The gamma term to control the instance selection..
	 */
	private DoubleField gammaField;

	/**
	 * Checkbox for variable tracking.
	 */
	private Checkbox variableTrackingCheckbox;

	/**
	 * Checkbox for variable tracking.
	 */
	private Checkbox processTrackingCheckbox;

	/**
	 * Result output to file checkbox.
	 */
	private Checkbox fileOutputCheckbox;

	/**
	 * The message area.
	 */
	private TextArea messageTextArea;

	/**
	 * How many times to repeat.
	 */
	private IntegerField repeatTimesField;

	/**
	 * The only constructor
	 */
	public SporGUI() {
		// A simple frame to contain the dialog
		Frame mainFrame = new Frame();
		mainFrame.setTitle("Self-Paced Co-Training for Regression");
		// The top part: select arff file.
		arffFilenameField = new FilenameField(30);
		arffFilenameField.setText("src/data/kin8nm.arff");
		Button browseButton = new Button(" Browse ");
		browseButton.addActionListener(arffFilenameField);

		Panel sourceFilePanel = new Panel();
		sourceFilePanel.add(new Label("The .arff file:"));
		sourceFilePanel.add(arffFilenameField);
		sourceFilePanel.add(browseButton);

		Panel algorithmPanel = new Panel();
		algorithmPanel.add(new Label("The learning measures:"));
		String[] algorithm = { "Spor", "Coreg", "Plain" };
		algorithmJComboBox = new JComboBox<String>(algorithm);
		algorithmJComboBox.setSelectedIndex(0);
		algorithmPanel.add(algorithmJComboBox);

		Panel numInstancePanel = new Panel();
		numInstancePanel.setLayout(new FlowLayout());
		labelFractionField = new DoubleField("0.05", 5);
		testFractionField = new DoubleField("0.3", 5);
		numInstancePanel.add(new Label("labeled instance rate:"));
		numInstancePanel.add(labelFractionField);
		numInstancePanel.add(new Label("unlabeled instance rate:"));
		numInstancePanel.add(testFractionField);
		labelFractionField.setEnabled(true);
		testFractionField.setEnabled(true);

		Panel tempProcessingPanel = new Panel();
		trainingIterations = new IntegerField("50");
		lamdaField = new DoubleField("1", 2);
		gammaField = new DoubleField("0.3", 2);
		stepSizeField = new DoubleField("0.005", 2);
		tempProcessingPanel.add(new Label("The training iterations:"));
		tempProcessingPanel.add(trainingIterations);
		tempProcessingPanel.add(new Label("The lambda:"));
		tempProcessingPanel.add(lamdaField);
		tempProcessingPanel.add(new Label("The gamma:"));
		tempProcessingPanel.add(gammaField);
		tempProcessingPanel.add(new Label("The stepsize:"));
		tempProcessingPanel.add(stepSizeField);

		Panel paraMetersPanel = new Panel();
		normalizeCheckbox = new Checkbox("normalize", true);
		disorderCheckbox = new Checkbox("disorder", true);
		lamdaCheckbox = new Checkbox("Add lamda", true);
		gammaCheckbox = new Checkbox("Add gamma", true);
		stepSizecheckbox = new Checkbox("Add stepsize", true);
		paraMetersPanel.add(lamdaCheckbox);
		paraMetersPanel.add(gammaCheckbox);
		paraMetersPanel.add(stepSizecheckbox);
		paraMetersPanel.add(normalizeCheckbox);
		paraMetersPanel.add(disorderCheckbox);

		Panel tempCotrianerLabel = new Panel();
		Label cotrainerLabel1 = new Label("The first cotrainer                     ");
		Label tempSpace = new Label("                             ");
		Label cotrainerLabel2 = new Label("                        The second cotrainer");
		tempCotrianerLabel.add(cotrainerLabel1);
		tempCotrianerLabel.add(tempSpace);
		tempCotrianerLabel.add(cotrainerLabel2);

		Panel distancePanel = new Panel();
		Label distanceLabel1 = new Label("Cotrainer1 distancemeasure");
		Label distanceLabel2 = new Label("Cotrainer2 distancemeasure");
		String[] distances = { "Euclidean", "Mahalanobis" };
		distanceJComboBox1 = new JComboBox<String>(distances);
		distanceJComboBox2 = new JComboBox<String>(distances);
		distanceJComboBox1.setSelectedIndex(0);
		distanceJComboBox2.setSelectedIndex(0);
		distancePanel.add(distanceLabel1);
		distancePanel.add(distanceJComboBox1);
		distancePanel.add(distanceLabel2);
		distancePanel.add(distanceJComboBox2);

		Panel knnNumberPanel = new Panel();
		Label knnNumberLabel1 = new Label("K(for Cotrainer1)");
		Label knnNumberLabel2 = new Label("K(for Cotrainer2)");
		knnNumberPanel.setLayout(new FlowLayout());
		kValueIntegerField1 = new IntegerField("3");
		kValueIntegerField2 = new IntegerField("3");
		knnNumberPanel.add(knnNumberLabel1);
		knnNumberPanel.add(kValueIntegerField1);
		knnNumberPanel.add(tempSpace);
		knnNumberPanel.add(knnNumberLabel2);
		knnNumberPanel.add(kValueIntegerField2);

		Panel processTrackingPanel = new Panel();
		processTrackingCheckbox = new Checkbox("Process tracking", false);
		variableTrackingCheckbox = new Checkbox("Variable tracking", false);
		fileOutputCheckbox = new Checkbox("Output to file", false);
		processTrackingPanel.add(processTrackingCheckbox);
		processTrackingPanel.add(variableTrackingCheckbox);
		processTrackingPanel.add(fileOutputCheckbox);

		Panel topPanel = new Panel();
		topPanel.setLayout(new GridLayout(10, 1));
		topPanel.add(sourceFilePanel);
		topPanel.add(algorithmPanel);
		topPanel.add(numInstancePanel);
		topPanel.add(tempProcessingPanel);
		topPanel.add(paraMetersPanel);
		topPanel.add(tempCotrianerLabel);
		topPanel.add(distancePanel);
		topPanel.add(knnNumberPanel);
		topPanel.add(processTrackingPanel);

		Panel centralPanel = new Panel();
		centralPanel.setLayout(new GridLayout(1, 1));
		messageTextArea = new TextArea(80, 40);
		centralPanel.add(messageTextArea);
		// The bottom part: ok and exit
		repeatTimesField = new IntegerField("20");
		Panel repeatTimesPanel = new Panel();
		repeatTimesPanel.add(new Label(" Repeat times: "));
		repeatTimesPanel.add(repeatTimesField);

		Button okButton = new Button(" OK ");
		okButton.addActionListener(this);
		// DialogCloser dialogCloser = new DialogCloser(this);
		Button exitButton = new Button(" Exit ");
		// cancelButton.addActionListener(dialogCloser);
		exitButton.addActionListener(ApplicationShutdown.applicationShutdown);
		Button helpButton = new Button(" Help ");
		helpButton.setSize(20, 10);
		HelpDialog helpDialog = null;
		try {
			helpDialog = new HelpDialog("Spor algorithm", "src/gui/SporgHelp.txt");
			helpButton.addActionListener(helpDialog);
		} catch (Exception ee) {
			try {
				helpDialog = new HelpDialog("Spor algorithm", "src/gui/SporgHelp.txt");
				helpButton.addActionListener(helpDialog);
			} catch (Exception ee2) {
				ErrorDialog.errorDialog.setMessageAndShow(ee.toString());
			} // Of try
		} // Of try
		Panel okPanel = new Panel();
		okPanel.add(okButton);
		okPanel.add(exitButton);
		okPanel.add(helpButton);

		Panel southPanel = new Panel();
		southPanel.setLayout(new GridLayout(2, 1));
		southPanel.add(repeatTimesPanel);
		southPanel.add(okPanel);

		mainFrame.setLayout(new BorderLayout());
		mainFrame.add(BorderLayout.NORTH, topPanel);
		mainFrame.add(BorderLayout.CENTER, centralPanel);
		mainFrame.add(BorderLayout.SOUTH, southPanel);

		mainFrame.setSize(700, 700);
		mainFrame.setLocation(10, 10);
		mainFrame.addWindowListener(ApplicationShutdown.applicationShutdown);
		mainFrame.setBackground(GUICommon.MY_COLOR);
		mainFrame.setVisible(true);

	}// Of the constructor

	/**
	 *************************** 
	 * The entrance method.
	 * 
	 * @param args The parameters.
	 *************************** 
	 */
	public static void main(String args[]) {
		new SporGUI();
	}// Of main

	/**
	 *************************** 
	 * Compare the results.
	 *************************** 
	 */
	public void actionPerformed(ActionEvent ae) {
		DecimalFormat decimalFormat = new DecimalFormat("0.000000000");
		messageTextArea.setText("Processing ... Please wait.\r\n");
		int tempRepeatTimes = repeatTimesField.getValue();

		// Parameters to be transferred to respective objects.
		int tempIterations = trainingIterations.getValue();
		String tempFilename = arffFilenameField.getText().trim();
		int tempCotrainer1DistanceMeasure = 0;
		int tempCotrainer2DistanceMeasure = 0;

		if (distanceJComboBox1.getSelectedIndex() == 0) {
			tempCotrainer1DistanceMeasure = DistanceMeasure.EUCLIDEAN;
		} else {
			tempCotrainer1DistanceMeasure = DistanceMeasure.MANHATTAN;
		} // Of if
		if (distanceJComboBox2.getSelectedIndex() == 0) {
			tempCotrainer2DistanceMeasure = DistanceMeasure.EUCLIDEAN;
		} else {
			tempCotrainer2DistanceMeasure = DistanceMeasure.MANHATTAN;
		} // Of if

		double tempStepSize = stepSizeField.getValue();
		boolean tempNormalize = normalizeCheckbox.getState();
		boolean tempDisorder = disorderCheckbox.getState();
		int tempCotrainerKValue1 = kValueIntegerField1.getValue();
		int tempCotrainerKValue2 = kValueIntegerField2.getValue();
		double lambda = lamdaField.getValue();
		
		String resultMessage = "";
		String tempString = "";
		double labelrate = labelFractionField.getValue();
		double tempError = 0;
		double gamma = gammaField.getValue();
		double maxError = 0;
		double minError = 0;
		double aveBeforeMse = 0;
		double aveAfterMse = 0;
		double wr1 = 0;
		double wr2 = 0;
		int wr3 = 0;
		int wr4 = 0;
		int firstAddeInstances = 0;
		int secondAddeInstances = 0;

		try {
			BufferedReader tempBufferedReader = new BufferedReader(new FileReader(tempFilename));
			Instances tempdata = new Instances(tempBufferedReader);
			tempBufferedReader.close();
			tempdata.setClassIndex(tempdata.numAttributes() - 1);
			if (tempNormalize == true) {
				SimpleTool.normalize(tempdata);
			} // Of if
			
			String tempDateString = SimpleTool.getTimeShort();
			File tempFile = new File("result" +"("+ tempDateString +")"+ ".txt");
			FileWriter out = new FileWriter(tempFile);
			
			for (int k = 0; k < tempRepeatTimes; k++) {
				messageTextArea.append("Round" + " " + k + " " + "complete" + "\r\n");
				if (tempDisorder == true) {
					SimpleTools.disorderData(tempdata);
				} // Of if

				Learner tempLearner = new Learner(tempdata, tempCotrainer1DistanceMeasure,
						tempCotrainer2DistanceMeasure, tempCotrainerKValue1, 100, tempCotrainerKValue2, labelrate,
						testFractionField.getValue(), tempIterations, lambda, tempStepSize, gamma);
				tempLearner.initializeParameters();

				if (algorithmJComboBox.getSelectedIndex() == 0) {
					if (lamdaCheckbox.getState() && stepSizecheckbox.getState() && gammaCheckbox.getState()) {
						tempLearner.spmcoCotraininges();
					} else if (lamdaCheckbox.getState() && stepSizecheckbox.getState() && !gammaCheckbox.getState()) {
						tempLearner.splCotraininges();
					} else if (lamdaCheckbox.getState() && !stepSizecheckbox.getState() && gammaCheckbox.getState()) {
						tempLearner.stableCotraininges();
					} else if (lamdaCheckbox.getState() && !stepSizecheckbox.getState() && !gammaCheckbox.getState()) {
						tempLearner.Cotraininges();
					} else {
						tempLearner.Cotraining();
					} // Of if
				} else if (algorithmJComboBox.getSelectedIndex() == 1) {
					tempLearner.Cotraining();
				} // Of if

				tempString = tempLearner.Learn(tempLearner.firstCotrainer, tempLearner.secondCotrainer);
				resultMessage += tempString + "\r\n";
				tempError += tempLearner.errorDrop;
				maxError = tempLearner.maxErrorDrop;
				minError = tempLearner.minErrorDrop;
				wr1 = tempLearner.errorDrop * 100;
				wr2 = tempLearner.afterMse;
				wr3 = tempLearner.firstCortaininerAddInstances;
				wr4 = tempLearner.secondCortaininerAddInstances;
				out.write(+wr1 + "\t");
				out.write(wr2 + "\t");
				out.write(wr3 + "\t");
				out.write(wr4 + "\r\n");
				firstAddeInstances += wr3;
				secondAddeInstances += wr4;
				aveBeforeMse += tempLearner.beforeMse;
				aveAfterMse += tempLearner.afterMse;
			} // Of for k
			out.close();
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try
		messageTextArea.append(resultMessage);
		messageTextArea.append("The max error dorp:" + decimalFormat.format(maxError) + "\r\n");
		messageTextArea.append("The min error dorp:" + decimalFormat.format(minError) + "\r\n");
		messageTextArea.append("The ave mean squared error before co-training:"
				+ decimalFormat.format(aveBeforeMse / tempRepeatTimes) + "\r\n");
		messageTextArea.append("The ave mean squared error after co-training:"
				+ decimalFormat.format(aveAfterMse / tempRepeatTimes) + "\r\n");
		messageTextArea.append(
				"The ave error dorp rate:" + decimalFormat.format((tempError / tempRepeatTimes) * 100) + "%" + "\r\n");
		messageTextArea.append("The average label1 added instance:" + firstAddeInstances / tempRepeatTimes + "\r\n");
		messageTextArea.append("The average label2 added instance:" + secondAddeInstances / tempRepeatTimes + "\r\n");
	}// Of actionPerformed
}// Of class SporGUI
