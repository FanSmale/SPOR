package gui;

import java.awt.*;
import java.awt.event.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Date;

import javax.swing.JComboBox;

import algorithm.*;
import common.Common;
import common.DistanceMeasure;
import common.SimpleTools;
import gui.guicommon.*;
import gui.guidialog.common.ErrorDialog;
import gui.guidialog.common.HelpDialog;
import gui.others.*;
import weka.core.Instances;

public class SporGUI implements ActionListener, ItemListener {
	/**
	 * Select the arff file.
	 */
	private FilenameField arffFilenameField;

	/**
	 * The proportion of labels that can be queried.
	 */
	private DoubleField labelFractionField;

	/**
	 * The proportion of representative labels queried in the first round.
	 */
	private DoubleField testFractionField;
	
	/**
	 * The iterations of training process.
	 */
	private IntegerField trainingIterations;

	/**
	 * Distance measures: Euclidean, Manhattan, Mahalanobis
	 */
	private JComboBox<String> distanceJComboBox1;

	/**
	 * Distance measures: Euclidean, Manhattan, Mahalanobis
	 */
	private JComboBox<String> distanceJComboBox2;

	/**
	 * Algorithm: SporGUI, Spor
	 */
	private JComboBox<String> algorithmJComboBox;

	/**
	 * Running with ultimate times or not.
	 */
	private Checkbox ultimatCheckbox;

	/**
	 * Get best number of K or not.
	 */
	private Checkbox crossValidCheckbox;

	private Checkbox stableTresholdCheckbox;

	/**
	 * Normalize or not.
	 */
	private Checkbox normalizeCheckbox;

	/**
	 * Self-pace learning or not.
	 */
	private Checkbox splCheckbox;

	/**
	 * AddTherehold in add instances or not.
	 */
	private Checkbox ThresholdCheckbox;

	/**
	 * AddTherehold in add instances or not.
	 */
	private DoubleField stableThresholdField;
	
	/**
	 * Disorder or not.
	 */
	private Checkbox disorderCheckbox;

	/**
	 * For density computation of Density Peaks (maybe also others.)
	 */
	private DoubleField adaptiveRatioDoubleField;

	/**
	 * Small block threshold. Small blocks will not be classified using the pure
	 * criteria.
	 */
	private IntegerField smallBlockThresholdIntegerField;

	/**
	 * The k value for kNN.
	 */
	private IntegerField kValueIntegerField1;
	/**
	 * The k value for kNN.
	 */
	private IntegerField kValueIntegerField2;

	/**
	 * For neighbor based weight as well as entropy computation.
	 */
	private DoubleField ThresholdField;
	
	/**
	 * For neighbor based weight as well as entropy computation.
	 */
	//private DoubleField stableThresholdField;

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
		mainFrame.setTitle("Semi-Supervised Regression with Co-Training Style Algorithm ");
		// The top part: select arff file.
		arffFilenameField = new FilenameField(30);
		arffFilenameField.setText("src/data/kin8nm.arff");
		Button browseButton = new Button(" Browse ");
		browseButton.addActionListener(arffFilenameField);

		Panel sourceFilePanel = new Panel();
		sourceFilePanel.add(new Label("The .arff file:"));
		sourceFilePanel.add(arffFilenameField);
		sourceFilePanel.add(browseButton);

		Panel numInstancePanel = new Panel();
		numInstancePanel.setLayout(new FlowLayout());
		labelFractionField = new DoubleField("0.01", 5);
		testFractionField = new DoubleField("0.3", 5);
		numInstancePanel.add(new Label("labeled instance rate:"));
		numInstancePanel.add(labelFractionField);
		numInstancePanel.add(new Label("unlabeled instance rate:"));
		numInstancePanel.add(testFractionField);
		labelFractionField.setEnabled(true);
		testFractionField.setEnabled(true);

		Panel tempProcessingPanel = new Panel();
		trainingIterations = new IntegerField("50");
		normalizeCheckbox = new Checkbox("normalize", true);
		disorderCheckbox = new Checkbox("disorder", true);
		ThresholdCheckbox = new Checkbox("Add Threshold", true);
		ThresholdField = new DoubleField("0.9");
		stableThresholdField = new DoubleField("0.025");
		stableTresholdCheckbox = new Checkbox("Stable Threshold", false);
		tempProcessingPanel.add(new Label("The training iterations:"));
		tempProcessingPanel.add(trainingIterations);
		tempProcessingPanel.add(new Label("The Threshold:"));
		tempProcessingPanel.add(ThresholdField);
		tempProcessingPanel.add(ThresholdCheckbox);
		tempProcessingPanel.add(normalizeCheckbox);
		tempProcessingPanel.add(disorderCheckbox);

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
		kValueIntegerField1 = new IntegerField("5");
		kValueIntegerField2 = new IntegerField("5");
		knnNumberPanel.add(knnNumberLabel1);
		knnNumberPanel.add(kValueIntegerField1);
		knnNumberPanel.add(tempSpace);
		knnNumberPanel.add(knnNumberLabel2);
		knnNumberPanel.add(kValueIntegerField2);

		processTrackingCheckbox = new Checkbox(" Process tracking ", false);
		variableTrackingCheckbox = new Checkbox(" Variable tracking ", false);
		fileOutputCheckbox = new Checkbox(" Output to file ", false);
		Panel trackingPanel = new Panel();
		trackingPanel.add(processTrackingCheckbox);
		trackingPanel.add(variableTrackingCheckbox);
		trackingPanel.add(fileOutputCheckbox);

		Panel topPanel = new Panel();
		topPanel.setLayout(new GridLayout(7, 1));
		topPanel.add(sourceFilePanel);
		topPanel.add(numInstancePanel);
		topPanel.add(tempProcessingPanel);
		topPanel.add(tempCotrianerLabel);
		topPanel.add(distancePanel);
		topPanel.add(knnNumberPanel);
		topPanel.add(trackingPanel);

		Panel centralPanel = new Panel();
		centralPanel.setLayout(new GridLayout(1, 1));
		messageTextArea = new TextArea(80, 40);
		centralPanel.add(messageTextArea);
		// The bottom part: ok and exit4
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
			helpDialog = new HelpDialog("SporGUI algorithm", "src/gui/SporHelp.txt");
			helpButton.addActionListener(helpDialog);
		} catch (Exception ee) {
			try {
				helpDialog = new HelpDialog("SporGUI algorithm", "gui/SporHelp.txt");
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

	}

	public static void main(String args[]) {
		new SporGUI();
	}// Of main

	/**
	 *************************** 
	 * Read the arff file.
	 *************************** 
	 */
	public void actionPerformed(ActionEvent ae) {
		SimpleTools.win = 0;
		SimpleTools.lose = 0;

		SimpleTools.NumInstances2added = 0;
		SimpleTools.NumInstances1added = 0;
		DecimalFormat df = new DecimalFormat("0.000000000");
		Common.startTime = new Date().getTime();
		messageTextArea.setText("Processing ... Please wait.\r\n");
		int tempRepeatTimes = repeatTimesField.getValue();
		int tempIterations = trainingIterations.getValue();
		String tempFilename = arffFilenameField.getText().trim();
		int tempCotrainer1DistanceMeasure = 0;
		int tempCotrainer2DistanceMeasure = 0;
		if (distanceJComboBox1.getSelectedIndex() == 0) {
			tempCotrainer1DistanceMeasure = DistanceMeasure.EUCLIDEAN;
		} else {
			tempCotrainer1DistanceMeasure = DistanceMeasure.MANHATTAN;
		}
		if (distanceJComboBox2.getSelectedIndex() == 0) {
			tempCotrainer2DistanceMeasure = DistanceMeasure.EUCLIDEAN;
		} else {
			tempCotrainer2DistanceMeasure = DistanceMeasure.MANHATTAN;
		}

		boolean tempNormalize = normalizeCheckbox.getState();

		boolean tempDisorder = disorderCheckbox.getState();
		int tempCotrainerKValue1 = kValueIntegerField1.getValue();
		int tempCotrainerKValue2 = kValueIntegerField2.getValue();
		String resultMessage = "";
		String tempString = "";
		double labelrate = labelFractionField.getValue();
		double wr1 = 0;
		double wr2 = 0;
		double wr3 = 0;
		int wr4 = 0;
		int wr5 = 0;
		try {
			BufferedReader r = new BufferedReader(new FileReader(tempFilename));
			Instances tempdata = new Instances(r);
			r.close();
			tempdata.setClassIndex(tempdata.numAttributes() - 1);
			if (tempNormalize == true) {
				// SimpleTools.normalizeDecisionSystem(tempdata);
				SimpleTool.normalize(tempdata);
			}

			double Threshold = ThresholdField.getValue();
			File file = new File("result.txt");
			FileWriter out = new FileWriter(file);

			for (int k = 0; k < tempRepeatTimes; k++) {
				SimpleTools.errorDrop = 0;
				SimpleTools.lastDrop = 0;
				SimpleTools.preDrop = 0;
				SimpleTools.NumInstances1added = 0;
				SimpleTools.NumInstances2added = 0;
				if (tempDisorder == true) {
					SimpleTools.disorderData(tempdata);
				}
				Learner tempLearner = new Learner(tempdata, tempCotrainer1DistanceMeasure,
						tempCotrainer2DistanceMeasure, tempCotrainerKValue1, 100,
						tempCotrainerKValue2, labelrate, testFractionField.getValue(),
						tempIterations, Threshold);
				if (crossValidCheckbox.getState() == true) {
					tempLearner.firstCotrainer.crossValid();
					tempLearner.secondCotrainer.crossValid();
				}
				
				if (ThresholdCheckbox.getState() == true
						&& stableTresholdCheckbox.getState() == false) {
					tempLearner.Cotraininges();
				}
				
				if (splCheckbox.getState() == false && ThresholdCheckbox.getState() == false
						&& stableTresholdCheckbox.getState() == false
						&& ultimatCheckbox.getState() == false) {
					tempLearner.cotraining();
				}
				tempLearner.firstCotrainer.denoise();
				// tempLearner.firstCotrainer.preDenoise();
				tempLearner.secondCotrainer.denoise();
				// tempLearner.secondCotrainer.preDenoise();
				tempString = tempLearner.Learn(tempLearner.firstCotrainer,
						tempLearner.secondCotrainer);
				resultMessage += tempString + "\r\n";
				wr1 = SimpleTools.errorDrop * 100;
				wr2 = SimpleTools.lastDrop * 100;
				wr3 = SimpleTools.preDrop;
				wr4 = (int) (SimpleTools.NumInstances1added);
				wr5 = (int) (SimpleTools.NumInstances2added);
				out.write(wr1 + "\t");
				out.write(wr2 + "\t");
				out.write(wr3 + "\t");
				out.write(wr4 + "\t");
				out.write(wr5 + "\r\n");
			}

			// }

			out.close();
		} catch (Exception e) {
			e.printStackTrace();

		}
		messageTextArea.append(resultMessage);
		messageTextArea.append("The denoising wins times:" + SimpleTools.win + "      "
				+ "The denoising loses times:" + SimpleTools.lose + "\r\n");
		messageTextArea
				.append("The max error dorp:" + df.format(SimpleTools.maxErrorDrop) + "\r\n");
		messageTextArea
				.append("The min error dorp:" + df.format(SimpleTools.minErrorDrop) + "\r\n");
		messageTextArea.append("The ave error dorp rate:"
				+ df.format((SimpleTools.errorDrop / tempRepeatTimes) * 100) + "%" + "\r\n");
		messageTextArea.append("The average label1 added instance:"
				+ SimpleTools.NumInstances1added / tempRepeatTimes + "\r\n");
		messageTextArea.append("The average label2 added instance:"
				+ SimpleTools.NumInstances2added / tempRepeatTimes + "\r\n");
	}

	@Override
	public void itemStateChanged(ItemEvent e) {
		// TODO Auto-generated method stub

	}

}
