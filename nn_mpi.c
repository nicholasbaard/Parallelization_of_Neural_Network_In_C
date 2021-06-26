#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ROW 773
#define COL 27

// Activation Functions
def sigmoid(double x){
    return 1/(1 + exp(-x));
}

def forward_propagate(double *input, double *weight1, double *weight2){
    for(int i = 0; i < hidden_nodesr; i++){
        for(int j = 0; j < input_nodes; j++){
            act += input[j]*weight[j][i];
        }
        layer1[i] = sigmoid(act);
    }
    for(int i = 0; i < output_nodes; i++){
        for(int j = 0; j < hidden_nodes; j++){
            act += input[j]*weight[j][i];
        }
        layer2[i] = sigmoid(act);
    }
}

int main(int argc, char *argv[]){
  float arr[ROW*COL];
  FILE* str = fopen("audit_risk_raw.csv", "r");

  // read in .csv data
  char line[1024];
  int count = 0;
  while (fgets(line, 1024, str))
  {
    char* tmp = strdup(line);
    char* c = strtok(tmp,",");

    while(c != NULL){
      arr[count] = atof(c);
      count ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }

    // NEURAL NETWORK

    // define nodes
    int input_nodes = 26;
    int hidden_nodes = 13;
    int output_nodes = 1;

    // define weights and biases
    double weight_layer1[hidden_nodes][input_nodes];
    double weight_layer2[output_nodes][hidden_nodes];

    double bias_layer1[hidden_nodes];
    double bias_layer2[output_nodes];

    // define layers of the NN to store the values
    double layer1[hidden_nodes];
    double layer2[output_nodes];

    // generate random weights and biases
    for(int i = 0; i < hidden_nodes; i++{
        for(int j = 0; j < input_nodes; j++){
            weight_layer1[i][j] = ((double)rand())/((double)RAND_MAX);
        }
    }
    for(int i = 0; i < output_nodes; i++){
        for(int j = 0; j < hidden_nodes; j++){
            weight_layer2[i][j] = ((double)rand())/((double)RAND_MAX);
        }
    }


  // print the array
  for (size_t i = 0; i < ROW; i++) {
    for (size_t j = 0; j < COL; j++) {
      printf("%f ", arr[i*COL + j]);
    }
    printf("\n");
  }
  return 0;
}
