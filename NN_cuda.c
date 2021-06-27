#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ROW 773
#define COL 26
#define TRAIN_ROW 541
#define TEST_ROW 232
// define nodes
#define INPUT_NODES 26
#define HIDDEN_NODES 10
#define OUTPUT_NODES 1


// Activation Functions
float sigmoid(float x){
    return 1/(1 + exp(-x));
}
float diff_Sigmoid(float x){
  return x * (1 - x);
}

float forward_propagate(float input[], float label[], float weight1[HIDDEN_NODES][INPUT_NODES], float weight2[OUTPUT_NODES][HIDDEN_NODES], float layer1[], float layer2[]){
    float act = 0;

    for(int i = 0; i < HIDDEN_NODES; i++){
        for(int j = 0; j < INPUT_NODES; j++){
            act += input[j]*weight1[i][j];
        }
        layer1[i] = sigmoid(act);
    }
    for(int i = 0; i < OUTPUT_NODES; i++){
        for(int j = 0; j < HIDDEN_NODES; j++){
            act += input[j]*weight2[i][j];
        }
        layer2[i] = sigmoid(act);
    }
    // BACKPROPAGATION:
    float d3[OUTPUT_NODES];
    for(int i = 0; i < OUTPUT_NODES; i++){
      //float error =
    }
    return 0;
}

int main(int argc, char *argv[]){
  //IMPORT TRAINING DATA
  float train_arr[TRAIN_ROW*COL];
  float train_mat[TRAIN_ROW][COL];
  FILE* str = fopen("train_data.csv", "r");

  char line[1024];
  int count = 0;
  while (fgets(line, 1024, str))
  {
    char* tmp = strdup(line);
    char* c = strtok(tmp,",");

    while(c != NULL){
      train_arr[count] = atof(c);
      count ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }
  // reshape arr into matrix
  for (size_t i = 0; i < TRAIN_ROW; i++) {
    for (size_t j = 0; j < COL; j++) {
      train_mat[i][j] = train_arr[i*COL + j];
      //printf("%f ", train_mat[i][j]);

    }
    //printf("\n");
  }
  //IMPORT TRAINING LABELS
  float train_y_arr[TRAIN_ROW*1];
  float train_y_mat[TRAIN_ROW][1];
  FILE* str_y = fopen("train_y.csv", "r");

  char line_y[1024];
  int count_y = 0;
  while (fgets(line_y, 1024, str_y))
  {
    char* tmp = strdup(line_y);
    char* c = strtok(tmp,",");

    while(c != NULL){
      train_y_arr[count_y] = atof(c);
      count_y ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }
  // reshape arr into matrix
  for (size_t i = 0; i < TRAIN_ROW; i++) {
    for (size_t j = 0; j < 1; j++) {
      train_y_mat[i][j] = train_y_arr[i*1 + j];
      //printf("%f ", train_y_mat[i][j]);

    }
    //printf("\n");
  }

  //IMPORT TESTING DATA
  float test_arr[TEST_ROW*COL];
  float test_mat[TEST_ROW][COL];
  FILE* str_t = fopen("test_data.csv", "r");

  char line_t[1024];
  int count_t = 0;
  while (fgets(line_t, 1024, str_t))
  {
    char* tmp = strdup(line_t);
    char* c = strtok(tmp,",");

    while(c != NULL){
      test_arr[count_t] = atof(c);
      count_t ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }
  // reshape arr into matrix
  for (size_t i = 0; i < TEST_ROW; i++) {
    for (size_t j = 0; j < COL; j++) {
      test_mat[i][j] = test_arr[i*COL + j];
      //printf("%f ", test_mat[i][j]);

    }
    //printf("\n");
  }
  //IMPORT TEST LABELS
  float test_y_arr[TEST_ROW*1];
  float test_y_mat[TEST_ROW][1];
  FILE* str_ty = fopen("test_y.csv", "r");

  char line_ty[1024];
  int count_ty = 0;
  while (fgets(line_ty, 1024, str_ty))
  {
    char* tmp = strdup(line_ty);
    char* c = strtok(tmp,",");

    while(c != NULL){
      test_y_arr[count_ty] = atof(c);
      count_ty ++;
      c = strtok(NULL, ",");
    }
    free(tmp);
  }
  // reshape arr into matrix
  for (size_t i = 0; i < TEST_ROW; i++) {
    for (size_t j = 0; j < 1; j++) {
      test_y_mat[i][j] = test_y_arr[i*1 + j];
      //printf("%f ", test_y_mat[i][j]);

    }
    //printf("\n");
  }

  // NEURAL NETWORK
  // define weights and biases
  float weight_layer1[HIDDEN_NODES][INPUT_NODES];
  float weight_layer2[OUTPUT_NODES][HIDDEN_NODES];

  float bias_layer1[HIDDEN_NODES];
  float bias_layer2[OUTPUT_NODES];

  // define layers of the NN to store the values
  float layer1[HIDDEN_NODES];
  float layer2[OUTPUT_NODES];


  // generate random weights and biases
  printf("%s\n", "Weight layer 1:");
  for(int i = 0; i < HIDDEN_NODES; i++){
      for(int j = 0; j < INPUT_NODES; j++){
          weight_layer1[i][j] = ((double)rand())/((double)RAND_MAX);
          printf("%f ", weight_layer1[i][j]);
      }
      printf("\n");
  }
  printf("%s\n", "Weight layer 2:");
  for(int i = 0; i < OUTPUT_NODES; i++){
      for(int j = 0; j < HIDDEN_NODES; j++){
          weight_layer2[i][j] = ((double)rand())/((double)RAND_MAX);
          printf("%f ", weight_layer2[i][j]);
      }
      printf("\n");
  }

  return 0;
}
