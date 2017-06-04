import java.security.SecureRandom;

//입력층 - 은닉 1층 - 은닉 2층 - 은닉 3층 - 출력층  으로 구성된 인공신경망입니다.

public class NeuralNetwork {
	int num_input;
	int num_hidden;
	int num_output;

	double[][] hidden1_weight;
	double[][] hidden2_weight;
	double[][] hidden3_weight;
	double[][] output_weight;

	void Compute_Output(double input[], double hidden1[], double hidden2[], double hidden3[], double output[]) {
		// hidden1
		for (int i = 0; i < num_hidden; i++) {
			double sum = 0;

			for (int j = 0; j < num_input; j++) {
				sum += input[j] * hidden1_weight[i][j];
			}
			sum += hidden1_weight[i][num_input];
			hidden1[i] = 1 / (1 + Math.exp(-sum));
		}
		// hidden2
		for (int i = 0; i < num_hidden; i++) {
			double sum = 0;

			for (int j = 0; j < num_hidden; j++) {
				sum += hidden1[j] * hidden2_weight[i][j];
			}
			sum += hidden2_weight[i][num_hidden];
			hidden2[i] = 1 / (1 + Math.exp(-sum));
		}
		// hidden3
		for (int i = 0; i < num_hidden; i++) {
			double sum = 0;

			for (int j = 0; j < num_hidden; j++) {
				sum += hidden2[j] * hidden3_weight[i][j];
			}
			sum += hidden3_weight[i][num_hidden];
			hidden3[i] = 1 / (1 + Math.exp(-sum));
		}
		// output
		for (int i = 0; i < num_output; i++) {
			double sum = 0;

			for (int j = 0; j < num_hidden; j++) {
				sum += hidden3[j] * output_weight[i][j];
			}
			sum += output_weight[i][num_hidden];
			output[i] = 1 / (1 + Math.exp(-sum));
		}
	}

	NeuralNetwork(int num_input, int num_hidden, int num_output) {
		this.num_input = num_input;
		this.num_hidden = num_hidden;
		this.num_output = num_output;

		output_weight = new double[num_output][];
		for (int i = 0; i < num_output; i++) {
			output_weight[i] = new double[num_hidden + 1];
		}
		hidden3_weight = new double[num_hidden][];
		for (int i = 0; i < num_hidden; i++) {
			hidden3_weight[i] = new double[num_hidden + 1];
		}
		hidden2_weight = new double[num_hidden][];
		for (int i = 0; i < num_hidden; i++) {
			hidden2_weight[i] = new double[num_hidden + 1];
		}
		hidden1_weight = new double[num_hidden][];
		for (int i = 0; i < num_hidden; i++) {
			hidden1_weight[i] = new double[num_input + 1];
		}

	}

	void Test(double input[], double output[]) {
		double[] hidden1 = new double[num_hidden];
		double[] hidden2 = new double[num_hidden];
		double[] hidden3 = new double[num_hidden];

		Compute_Output(input, hidden1, hidden2, hidden3, output);

	}

	void Train(int num_train, double learning_rate, double[][] input, double[][] target_output) {
		int num_epoch = 0;
		int max_epoch = 100000;

		double[] hidden1 = new double[num_hidden];
		double[] hidden1_derivative = new double[num_hidden];
		double[] hidden2 = new double[num_hidden];
		double[] hidden2_derivative = new double[num_hidden];
		double[] hidden3 = new double[num_hidden];
		double[] hidden3_derivative = new double[num_hidden];
		double[] output = new double[num_output];
		double[] output_derivative = new double[num_output];

		SecureRandom random = new SecureRandom();

		for (int i = 0; i < num_hidden; i++) {
			for (int j = 0; j < num_input + 1; j++) {
				hidden1_weight[i][j] = 0.2 * random.nextDouble();
			}
		}
		for (int i = 0; i < num_hidden; i++) {
			for (int j = 0; j < num_hidden + 1; j++) {
				hidden2_weight[i][j] = 0.2 * random.nextDouble();
			}
		}
		for (int i = 0; i < num_hidden; i++) {
			for (int j = 0; j < num_hidden + 1; j++) {
				hidden3_weight[i][j] = 0.2 * random.nextDouble();
			}
		}
		for (int i = 0; i < num_output; i++) {
			for (int j = 0; j < num_hidden + 1; j++) {
				output_weight[i][j] = 0.2 * random.nextDouble();
			}
		}

		do {
			double error = 0;

			for (int i = 0; i < num_train; i++) {
				Compute_Output(input[i], hidden1, hidden2, hidden3, output);

				// 출력미분값 계산
				for (int j = 0; j < num_output; j++) {
					output_derivative[j] = learning_rate * (output[j] - target_output[i][j]) * (1 - output[j])
							* output[j];
				}

				// 출력가중치 조정
				for (int j = 0; j < num_output; j++) {
					for (int k = 0; k < num_hidden; k++) {
						output_weight[j][k] -= output_derivative[j] * hidden3[k];
					}
					output_weight[j][num_hidden] -= output_derivative[j];
				}
				//////////////////////////////////////////////////////////////////////////
				// 은닉3미분값 계산
				for (int j = 0; j < num_hidden; j++) {
					double sum = 0;

					for (int k = 0; k < num_output; k++) {
						sum += output_derivative[k] * output_weight[k][j];
					}
					hidden3_derivative[j] = sum * (1 - hidden3[j]) * hidden3[j];
				}

				// 은닉3가중치 조정
				for (int j = 0; j < num_hidden; j++) {
					for (int k = 0; k < num_hidden; k++) {
						hidden3_weight[j][k] -= hidden3_derivative[j] * hidden2[k];
					}
					hidden3_weight[j][num_hidden] -= hidden3_derivative[j];
				}
				//////////////////////////////////////////////////////////////////////////
				// 은닉2미분값 계산
				for (int j = 0; j < num_hidden; j++) {
					double sum = 0;

					for (int k = 0; k < num_hidden; k++) {
						sum += hidden3_derivative[k] * hidden3_weight[k][j];
					}
					hidden2_derivative[j] = sum * (1 - hidden2[j]) * hidden2[j];
				}

				// 은닉2가중치 조정
				for (int j = 0; j < num_hidden; j++) {
					for (int k = 0; k < num_hidden; k++) {
						hidden2_weight[j][k] -= hidden2_derivative[j] * hidden1[k];
					}
					hidden2_weight[j][num_hidden] -= hidden2_derivative[j];
				}
				//////////////////////////////////////////////////////////////////////////
				// 은닉1미분값 계산
				for (int j = 0; j < num_hidden; j++) {
					double sum = 0;

					for (int k = 0; k < num_hidden; k++) {
						sum += hidden2_derivative[k] * hidden2_weight[k][j];
					}
					hidden1_derivative[j] = sum * (1 - hidden1[j]) * hidden1[j];
				}

				// 은닉1가중치 조정
				for (int j = 0; j < num_hidden; j++) {
					for (int k = 0; k < num_input; k++) {
						hidden1_weight[j][k] -= hidden1_derivative[j] * input[i][k];
					}
					hidden1_weight[j][num_input] -= hidden1_derivative[j];
				}

				// 오차 계산
				for (int j = 0; j < num_output; j++) {
					error += 0.5 * (output[j] - target_output[i][j]) * (output[j] - target_output[i][j]);
				}
			}
			if (num_epoch % 500 == 0) {
				System.out.println("Rep Count:" + num_epoch + ", Error: " + error);
			}
		} while (num_epoch++ < max_epoch);

	}
}
