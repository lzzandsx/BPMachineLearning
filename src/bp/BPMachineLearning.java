package bp;

public class BPMachineLearning {

	int input_layer_num = 3;
	int hidden_layer_num = 2;
	double[] input_layer_input;// 输入层输入和输出
	double[] hidden_layer_input;// 隐含层输入
	double[] hidden_layer_output;// 隐含层输出
	double[][] input_layer_w;// 输入层对隐含层的权值
	double[][] hidden_layer_v;// 隐含层对输出层的权值
	double[] panzhi;
	double[] panzhi_w;
	double output_layer_input;
	double output_layer_output;
	double stand_output;
	double output_error_rate;
	double study_rate;
	double error = 1;// 总误差

	/**
	 * 输入层三个节点、隐含层两个节点、输出层一个节点 学习系数为1，偏值为1
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO 自动生成的方法存根
		BPMachineLearning bp = new BPMachineLearning();
		bp.execute();
	}

	public void execute() {
		init();

		while (error>(1e-6)) {
			forward_propagation();
			afterward_propagation();
			System.out.println("output "+output_layer_output);
			System.out.println("error "+error);
			System.out.println();
		}
	}

	public void init() {
		// 输入
		input_layer_input = new double[input_layer_num];
		for (int i = 0; i < input_layer_num; i++) {
			input_layer_input[i] = i + 1;
		}
		// 偏值
		panzhi = new double[2];
		panzhi[0] = 1;
		panzhi[1] = 1;
		// 偏值得权值
		panzhi_w = new double[2];
		panzhi_w[0] = 0.6;
		panzhi_w[1] = 0.4;
		// 输入层权值
		input_layer_w = new double[input_layer_num][hidden_layer_num];
		System.out.println("输入层权值");
		for (int i = 0; i < input_layer_num; i++) {
			for (int j = 0; j < hidden_layer_num; j++) {
				double temp = (int) (Math.random() * 100) / 100.0;
				input_layer_w[i][j] = temp;
				System.out.println(input_layer_w[i][j]);
			}
		}
		// 初始化隐含层输入
		hidden_layer_input = new double[hidden_layer_num];
		// 初始化隐含层输出
		hidden_layer_output = new double[hidden_layer_num];

		// 隐含层权值
		hidden_layer_v = new double[hidden_layer_num][1];
		System.out.println("隐含层权值");
		for (int i = 0; i < hidden_layer_num; i++) {
			for (int j = 0; j < 1; j++) {
				double temp = (int) (Math.random() * 100) / 100.0;
				hidden_layer_v[i][j] = temp;
				System.out.println(hidden_layer_v[i][j]);
			}
		}
		// 标准输出
		stand_output = 0.6;
		// 学习率
		study_rate = 0.5;

	}

	public void forward_propagation() {

		for (int i = 0; i < hidden_layer_num; i++) {
			hidden_layer_input[i] = 0;
			for (int j = 0; j < input_layer_num; j++) {
				hidden_layer_input[i] += input_layer_input[j] * input_layer_w[j][i];
			}
			hidden_layer_input[i] += panzhi[0] * panzhi_w[0];
			hidden_layer_output[i] = sigmoid_function(hidden_layer_input[i]);
		}

		output_layer_input = 0;
		for (int i = 0; i < hidden_layer_num; i++) {
			output_layer_input += hidden_layer_output[i] * hidden_layer_v[i][0];
		}
		output_layer_input += panzhi[1] * panzhi_w[1];
		output_layer_output = sigmoid_function(output_layer_input);

	}

	public void afterward_propagation() {

		error = Math.pow(stand_output - output_layer_output, 2) / 2;
		output_error_rate = -(stand_output - output_layer_output) * (output_layer_output) * (1 - output_layer_output);

		// 更新输入层权值
		for (int i = 0; i < input_layer_num; i++) {
			for (int j = 0; j < hidden_layer_num; j++) {
				double rate = input_layer_input[i] * hidden_layer_output[j]* (1-hidden_layer_output[j]) * output_error_rate * hidden_layer_v[j][0];
				input_layer_w[i][j] = input_layer_w[i][j] - study_rate * rate;
			}
		}

		// 更新隐含层权值
		for (int i = 0; i < hidden_layer_num; i++) {
			hidden_layer_v[i][0] = hidden_layer_v[i][0] - study_rate * hidden_layer_output[i] * output_error_rate;
		}

	}

	public double sigmoid_function(double input) {
		return 1 / (1 + Math.pow(Math.E, -input));
	}

	public double p_sigmoid_function(double input) {
		return sigmoid_function(input) * (1 - sigmoid_function(input));
	}

}
