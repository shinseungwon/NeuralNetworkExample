public class MainClass {

	public static final int BOARD_WIDTH = 16;

	public static void main(String[] args) {

		int num_input = 8;
		int num_hidden = 32;
		int num_output = 2;
		int num_train = 4;

		double learning_rate = 0.1;

		double[][] input = new double[num_train][];
		double[][] target_output = new double[num_train][];

		NeuralNetwork FNN = new NeuralNetwork(num_input, num_hidden, num_output);

		for (int i = 0; i < num_train; i++) {
			input[i] = new double[num_input];
			target_output[i] = new double[num_output];
		}

		input[0][0] = 0;
		input[0][1] = 1;
		input[0][2] = 0;
		input[0][3] = 0;
		input[0][4] = 1;
		input[0][5] = 1;
		input[0][6] = 1;
		input[0][7] = 0;

		input[1][0] = 0;
		input[1][1] = 1;
		input[1][2] = 0;
		input[1][3] = 1;
		input[1][4] = 0;
		input[1][5] = 1;
		input[1][6] = 0;
		input[1][7] = 1;

		input[2][0] = 1;
		input[2][1] = 1;
		input[2][2] = 0;
		input[2][3] = 1;
		input[2][4] = 1;
		input[2][5] = 1;
		input[2][6] = 1;
		input[2][7] = 1;

		input[3][0] = 0;
		input[3][1] = 0;
		input[3][2] = 0;
		input[3][3] = 0;
		input[3][4] = 1;
		input[3][5] = 1;
		input[3][6] = 1;
		input[3][7] = 1;

		target_output[0][0] = 0.3;
		target_output[0][1] = 0.6;
		target_output[1][0] = 0.1;
		target_output[1][1] = 0.9;
		target_output[2][0] = 0.01;
		target_output[2][1] = 1;
		target_output[3][0] = 0.09;
		target_output[3][1] = 0.24;

		FNN.Train(num_train, learning_rate, input, target_output);
		System.out.println();
		for (int i = 0; i < num_train; i++) {
			double[] output = new double[num_output];

			System.out.print("Input: ");
			for (int j = 0; j < num_input; j++) {
				System.out.print(input[i][j] + " ");
			}
			FNN.Test(input[i], output);
			System.out.println();
			System.out.print("Output:");
			for (int j = 0; j < num_output; j++) {
				System.out.printf("%.5f ", output[j]);
			}
			System.out.println();
		}
		
		System.out.println("-----Test-----");

		double[] testInput = new double[num_input];
		double[] testOutput = new double[num_output];
		testInput[0] = 0;
		testInput[1] = 1;
		testInput[2] = 0;
		testInput[3] = 0;
		testInput[4] = 1;
		testInput[5] = 1;
		testInput[6] = 1;
		testInput[7] = 0.5;

		System.out.print("Input: ");
		for (int j = 0; j < num_input; j++) {
			System.out.print(testInput[j] + " ");
		}
		FNN.Test(testInput, testOutput);
		System.out.println();
		System.out.print("Output:");
		for (int j = 0; j < num_output; j++) {
			System.out.printf("%.5f ", testOutput[j]);
		}
		System.out.println();
	}

}
