#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 100
#define OUTPUT_SIZE 1

#pragma warn -rvl
#pragma warn -par
#pragma warn -rch
typedef struct {
  double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE];
  double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
  double biases_hidden[HIDDEN_SIZE];
  double biases_output[OUTPUT_SIZE];
} NeuralNetwork;

double tanh_activation(double x) { return tanh(x); }
void feedforward(NeuralNetwork *nn, double input[INPUT_SIZE],
                 double hidden[HIDDEN_SIZE], double output[OUTPUT_SIZE]);
void train(NeuralNetwork *nn, double input[INPUT_SIZE],
           double target[OUTPUT_SIZE], double learning_rate);

typedef struct {
  NeuralNetwork *nn;
  double input[INPUT_SIZE];
  double target[OUTPUT_SIZE];
  double learning_rate;
} ThreadArgs;
#pragma omp parallel for
void *run_training(void *arg) {
  ThreadArgs *a = (ThreadArgs *)arg;
  double error_threshold = 0.01;
  double error = 1.0;
  double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
  while (error > error_threshold) {
    train(a->nn, a->input, a->target, a->learning_rate);
    feedforward(a->nn, a->input, hidden, output);
    error = fabs(a->target[0] - output[0]);
    printf("Current output: %.6f, Error: %.6f\n", output[0], error);
  }
  printf("The training is finally complete. Here is the output: %.6f\n",
         output[0]);
  return NULL;
}

void initialize_network(NeuralNetwork *nn) {
  srand((unsigned int)time(NULL));
  for (int i = 0; i < INPUT_SIZE; ++i)
    for (int j = 0; j < HIDDEN_SIZE; ++j)
      nn->weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  for (int i = 0; i < HIDDEN_SIZE; ++i)
    for (int j = 0; j < OUTPUT_SIZE; ++j)
      nn->weights_hidden_output[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  for (int i = 0; i < HIDDEN_SIZE; ++i)
    nn->biases_hidden[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
  for (int i = 0; i < OUTPUT_SIZE; ++i)
    nn->biases_output[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

void feedforward(NeuralNetwork *nn, double input[INPUT_SIZE],
                 double hidden[HIDDEN_SIZE], double output[OUTPUT_SIZE]) {
  #pragma omp parallel for
  for (int i = 0; i < HIDDEN_SIZE; ++i) {
    hidden[i] = nn->biases_hidden[i];
    for (int j = 0; j < INPUT_SIZE; ++j)
      hidden[i] += input[j] * nn->weights_input_hidden[j][i];
    hidden[i] = tanh_activation(hidden[i]);
  }
  for (int i = 0; i < OUTPUT_SIZE; ++i) {
    output[i] = nn->biases_output[i];
    for (int j = 0; j < HIDDEN_SIZE; ++j)
      output[i] += hidden[j] * nn->weights_hidden_output[j][i];
    output[i] = tanh_activation(output[i]);
  }
}

void train(NeuralNetwork *nn, double input[INPUT_SIZE],
           double target[OUTPUT_SIZE], double learning_rate) {
  double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
  feedforward(nn, input, hidden, output);

  double output_errors[OUTPUT_SIZE];
  for (int i = 0; i < OUTPUT_SIZE; ++i)
    output_errors[i] = target[i] - output[i];

  for (int i = 0; i < OUTPUT_SIZE; ++i) {
    for (int j = 0; j < HIDDEN_SIZE; ++j)
      nn->weights_hidden_output[j][i] += learning_rate * output_errors[i] *
                                         (1 - output[i] * output[i]) *
                                         hidden[j];
    nn->biases_output[i] +=
        learning_rate * output_errors[i] * (1 - output[i] * output[i]);
  }

  #pragma omp parallel for
  for (int i = 0; i < HIDDEN_SIZE; ++i) {
    double hidden_error = 0.0;
    for (int j = 0; j < OUTPUT_SIZE; ++j)
      hidden_error += output_errors[j] * nn->weights_hidden_output[i][j];
    for (int j = 0; j < INPUT_SIZE; ++j)
      nn->weights_input_hidden[j][i] +=
          learning_rate * hidden_error * (1 - hidden[i] * hidden[i]) * input[j];
    nn->biases_hidden[i] +=
        learning_rate * hidden_error * (1 - hidden[i] * hidden[i]);
  }
}

int main(int argc, char *argv[]) {

	if (argc < 3){
		printf("Wrong command son, command : %s <target value>, <learning rate> \n", argv[0]);
		return 1;
	}
  NeuralNetwork nn;
  initialize_network(&nn);
  ThreadArgs args;
  args.nn = &nn;
  args.input[0] = 0.5;
  args.input[1] = 0.3;
  args.target[0] = strtod(argv[1], NULL);
  args.learning_rate = strtod(argv[2], NULL);
  pthread_t thread;
  pthread_create(&thread, NULL, run_training, &args);
  pthread_join(thread, NULL);
  return 0;
}
