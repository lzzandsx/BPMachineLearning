package bp;

import java.util.Vector;

public class BPextend {

	int input_layer_num = 3;
	int hidden_layer_num = 3;
	int output_layer_num = 2;
	double[] input_layer_input;// 输入层输入和输出
	double[] hidden_layer_input;// 隐含层输入
	double[] hidden_layer_output;// 隐含层输出
	double[] output_layer_input;//输出层输入
	double[] output_layer_output;//输出层输出
	double[] output_error_rate;//输出层各自节点的误差率
	double[][] input_layer_w;// 输入层对隐含层的权值
	double[][] hidden_layer_v;// 隐含层对输出层的权值
	double[] panzhi;
	double[] panzhi_w;	
	double[] stand_output;	
	double study_rate; //学习率
	double total_error = 1;// 总误差,让第一次能运行而初始化先
	
	public static void main(String[] args) {
		// TODO 自动生成的方法存根
		BPextend bPextend = new BPextend();
		bPextend.execute();
	}

	public void execute() {
		init();
		while (total_error>(1e-6)) {
			forward_propagation();
			afterward_propagation();
		}
		System.out.println("error "+total_error);
		for(int i=0;i<output_layer_num;i++){
			System.out.println(output_layer_output[i]);
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
		hidden_layer_v = new double[hidden_layer_num][output_layer_num];
		System.out.println("隐含层权值");
		for (int i = 0; i < hidden_layer_num; i++) {
			for (int j = 0; j < output_layer_num; j++) {
				double temp = (int) (Math.random() * 100) / 100.0;
				hidden_layer_v[i][j] = temp;
				System.out.println(hidden_layer_v[i][j]);
			}
		}
		// 标准输出
		stand_output = new double[output_layer_num];
		for(int i=0;i<output_layer_num;i++){
			stand_output[i] = 0.6;
		}		
		// 初始化输出层
		output_layer_input = new double[output_layer_num];
		output_layer_output = new double[output_layer_num];
		output_error_rate = new double[output_layer_num];
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

		for(int i=0;i< output_layer_num;i++){
			output_layer_input[i] = 0;
			for (int j = 0; j < hidden_layer_num; j++) {
				output_layer_input[i] += hidden_layer_output[j] * hidden_layer_v[j][i];
			}
			output_layer_input[i] += panzhi[1] * panzhi_w[1];
			output_layer_output[i] = sigmoid_function(output_layer_input[i]);
		}
	}

	public void afterward_propagation() {

		total_error = 0;
		for(int i=0;i<output_layer_num;i++){
			output_error_rate[i] = -(stand_output[i] - output_layer_output[i]) * (output_layer_output[i]) * (1 - output_layer_output[i]);
			total_error += Math.pow(stand_output[i] - output_layer_output[i], 2) / 2;
		}
		
		
		
		// 更新输入层权值
		for (int i = 0; i < input_layer_num; i++) {
			for (int j = 0; j < hidden_layer_num; j++) {
				double rate = 0;
				for(int k=0;k<output_layer_num;k++){
					rate +=  output_error_rate[k] * hidden_layer_v[j][k];
					
				}
				rate *= input_layer_input[i] * hidden_layer_output[j]* (1-hidden_layer_output[j]) ;
				input_layer_w[i][j] = input_layer_w[i][j] - study_rate * rate;
			}
		}

		// 更新隐含层权值
		for (int i = 0; i < hidden_layer_num; i++) {
			for(int j=0;j< output_layer_num;j++){
				hidden_layer_v[i][j] = hidden_layer_v[i][j] - study_rate * hidden_layer_output[i] * output_error_rate[j];
			}
		}

	}

	public double sigmoid_function(double input) {
		return 1 / (1 + Math.pow(Math.E, -input));
	}

	public double p_sigmoid_function(double input) {
		return sigmoid_function(input) * (1 - sigmoid_function(input));
	}

	
	
}
