package algorithm;

import java.util.Arrays;

/**
 * A toy example. <br>
 * Project: Self-paced learning.<br>
 * 
 * @author Fan Min<br>
 *         www.fansmale.com, github.com/fansmale/MFAdaBoosting.<br>
 *         Email: minfan@swpu.edu.cn, minfanphd@163.com.<br>
 *         Data Created: July 26, 2020.<br>
 *         Last modified: July 26, 2020.
 * @version 1.0
 */

public class ToyExample {

	/**
	 ****************** 
	 * For integration test.
	 * 
	 * @param args
	 *            Not provided.
	 ****************** 
	 */
	public static int[] spld(double[] loss, int[] groupmembership,
			double lambda, double gamma) {
		// Step 1. Count the number of groups. We assume that the group number
		// is from 1 to k.
		int tempNumGroups = -1;
		for (int i = 0; i < groupmembership.length; i++) {
			if (tempNumGroups < groupmembership[i]) {
				tempNumGroups = groupmembership[i];
			}// Of if
		}// Of for i
		System.out.println("tempNumGroups = " + tempNumGroups);

		int[] groupidx = new int[tempNumGroups];
		for (int i = 0; i < groupidx.length; i++) {
			groupidx[i] = i + 1;
		}// Of for i
		System.out.println("groupidx = " + Arrays.toString(groupidx));
		// R: groupidx = unique(groupmembership)

		// Step 2. Initialize.
		boolean[] selectedidx = new boolean[loss.length];
		double[] selectedscores = new double[loss.length];

		// Step 3.
		for (int i = 0; i < tempNumGroups; i++) {
			// Step 3.1 The idx_ingroup array.
			int[] idx_ingroup = which(groupmembership, groupidx[i]);
			System.out.println("idx_ingroup = " + Arrays.toString(idx_ingroup));

			// Step 3.2 The loss_ingroup array
			double[] loss_ingroup = new double[idx_ingroup.length];
			for (int j = 0; j < loss_ingroup.length; j++) {
				loss_ingroup[j] = loss[idx_ingroup[j]];
			}// Of for j

			// Step 3.3 The rank_ingroup array
			System.out.println("loss_ingroup = "
					+ Arrays.toString(loss_ingroup));
			int[] rank_ingroup = rankDesendant(loss_ingroup);
			System.out.println("rank_ingroup = "
					+ Arrays.toString(rank_ingroup));

			int nj = idx_ingroup.length;

			for (int j = 0; j < nj; j++) {
				if (loss_ingroup[j] < lambda
						+ gamma
						/ (Math.sqrt(rank_ingroup[j]) + Math
								.sqrt(rank_ingroup[j] - 1))) {
					selectedidx[idx_ingroup[j]] = true;
					System.out.println("Selecting: " + idx_ingroup[j]);
				} else {
					selectedidx[idx_ingroup[j]] = false;
					// System.out.println("deselecting: " + idx_ingroup[j]);
				}// Of if
				selectedscores[idx_ingroup[j]] = loss_ingroup[j]
						- lambda
						- gamma
						/ (Math.sqrt(rank_ingroup[j]) + Math
								.sqrt(rank_ingroup[j] - 1));
			}// Of for j
		}// Of for i

		int[] newSelectedIndices = which(selectedidx);
		// int[] sortedselectedidx =
		Arrays.sort(newSelectedIndices);

		/*
		 * groupidx = unique(groupmembership) b = length(groupidx) selectedidx =
		 * replicate(length(loss), 0) selectedsocres = replicate(length(loss),
		 * 0) #used to rank the samples at the end for(j in 1:b) { idx_ingroup
		 * <- which(groupmembership==groupidx[j]) loss_ingroup <-
		 * loss[idx_ingroup] rank_ingroup <- rank(loss_ingroup,
		 * ties.method="first") nj = length(idx_ingroup) for(i in 1:nj) {
		 * if(loss_ingroup[i] < lambda +
		 * gamma/(sqrt(rank_ingroup[i])+sqrt(rank_ingroup[i]-1))) {
		 * selectedidx[idx_ingroup[i]]=1 } else { selectedidx[idx_ingroup[i]]=0
		 * } selectedsocres[idx_ingroup[i]] = loss_ingroup[i] - lambda -
		 * gamma/(sqrt(rank_ingroup[i])+sqrt(rank_ingroup[i]-1)) } }
		 * 
		 * selectedidx = which(selectedidx == 1) sortedselectedidx =
		 * selectedidx[sort(selectedsocres[selectedidx], decreasing=FALSE,
		 * index.return=TRUE)$ix] #print(sort(selectedsocres[selectedidx]))
		 */

		System.out.println("Indices: " + Arrays.toString(newSelectedIndices));

		return newSelectedIndices;
	}// Of spld

	/**
	 ****************** 
	 * Compress an array.
	 * 
	 * @param paraArray
	 *            The given array.
	 ****************** 
	 */
	public static int[] compressArray(int[] paraArray, int paraLength) {
		int[] resultArray = new int[paraLength];
		for (int i = 0; i < resultArray.length; i++) {
			resultArray[i] = paraArray[i];
		}// Of for i
		return resultArray;
	}// Of compressArray

	/**
	 ****************** 
	 * Rank the array. The biggest value will be 1.
	 * 
	 * @param paraArray
	 *            The given array.
	 ****************** 
	 */
	public static int[] rankDesendant(double[] paraArray) {
		int[] resultArray = new int[paraArray.length];
		for (int i = 0; i < paraArray.length; i++) {
			resultArray[i] = 1;
			for (int j = 0; j < paraArray.length; j++) {
				if (paraArray[j] > paraArray[i]) {
					resultArray[i]++;
				}// Of if

				if (paraArray[j] == paraArray[i]) {
					if (j < i) {
						resultArray[i]++;
					}// Of if
				}// Of if
			}// Of for j
		}// Of for i
		return resultArray;
	}// Of compressArray

	/**
	 ****************** 
	 * Which elements in the array are equal to the given value.
	 * 
	 * @param paraArray
	 *            The given array.
	 ****************** 
	 */
	public static int[] which(int[] paraArray, int paraValue) {
		int[] tempArray = new int[paraArray.length];

		int tempCounter = 0;
		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i] == paraValue) {
				tempArray[tempCounter] = i;
				tempCounter++;
			}// Of if
		}// Of for i

		int[] resultArray = compressArray(tempArray, tempCounter);

		return resultArray;
	}// Of which

	/**
	 ****************** 
	 * Which elements in the array are true.
	 * 
	 * @param paraArray
	 *            The given array.
	 ****************** 
	 */
	public static int[] which(boolean[] paraArray) {
		int[] tempArray = new int[paraArray.length];

		int tempCounter = 0;
		for (int i = 0; i < paraArray.length; i++) {
			if (paraArray[i]) {
				tempArray[tempCounter] = i;
				tempCounter++;
			}// Of if
		}// Of for i

		int[] resultArray = compressArray(tempArray, tempCounter);

		return resultArray;
	}// Of which

	/**
	 ****************** 
	 * Obtain the subset of a char set
	 * 
	 * @param paraArray
	 *            The given char array.
	 ****************** 
	 */
	public static char[] charSubset(char[] paraArray, int[] paraIndices) {
		char[] resultArray = new char[paraIndices.length];
		for (int i = 0; i < paraIndices.length; i++) {
			resultArray[i] = paraArray[paraIndices[i]];
		}// Of for i

		return resultArray;
	}// Of charSubset

	/**
	 ****************** 
	 * For integration test.
	 * 
	 * @param args
	 *            Not provided.
	 ****************** 
	 */
	public static void main(String args[]) {
		char[] vid = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
				'l', 'm', 'n' };
		int[] groupmembership = { 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4 };
		double[] loss = { 0.05, 0.12, 0.12, 0.12, 0.15, 0.40, 0.17, 0.18, 0.35,
				0.15, 0.16, 0.20, 0.50, 0.28 };

		// System.out.println("When lambda=0.15, SPL selects:");
		// System.out.println(spl(loss, 0.15));
		int[] tempSelections = null;
		tempSelections = spld(loss, groupmembership, 0.05, 0.2);
		char[] tempSelectionsInChars = charSubset(vid, tempSelections);

		String tempResult = Arrays.toString(tempSelectionsInChars);
		System.out.println("*****************");
		System.out.println("When lambda = 0.03 and gamma = 0.2, SPLD selects: "
				+ tempResult);

		tempSelections = spld(loss, groupmembership, 0.00, 0.285);
		tempSelectionsInChars = charSubset(vid, tempSelections);
		tempResult = Arrays.toString(tempSelectionsInChars);
		System.out.println("*****************");
		System.out.println("When lambda = 0.0 and gamma = 0.2, SPLD selects: "
				+ tempResult);
	}// Of main

}// Of class ToyExample

