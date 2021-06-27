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

#define ALPHA 0.1

// Activation Functions
float sigmoid(float x){
    return 1/(1 + exp(-x));
}

float diff_Sigmoid(float x){
  return x * (1 - x);
}

float predict(float **input_matrix,
            float *pred_arr,
            float **weight1,
            float **weight2,
            float layer1[HIDDEN_NODES],
            float layer2[OUTPUT_NODES])
            {

    //this will be each extracted input row
    float input[COL];
    float output=0;
    // iterate through input matrix row by row, extracting each row for training
    for(int row = 0; row < TRAIN_ROW; row++){
        for(int col = 0; col < COL; col++){
            input[col] = input_matrix[row][col];
        }
        //printf("%s\n", "hey");
        //this is for one row instance of forward and backprop
        // FORWARD PROPAGATION:
        for(int i = 0; i < HIDDEN_NODES; i++){
            float act = 0.0;
            for(int j = 0; j < INPUT_NODES; j++){
                act += input[j]*weight1[i][j];
                //printf("%f\n", act);
            }
            layer1[i] = sigmoid(act);
        }
        for(int i = 0; i < OUTPUT_NODES; i++){
            float act = 0.0;
            for(int j = 0; j < HIDDEN_NODES; j++){
                act += layer1[j]*weight2[i][j];
                //printf("act: %f\n", act);
            }
            layer2[i] = sigmoid(act);
        }
        //printf("layer 2: %f\n", layer2[0]);
        // BACKPROPAGATION:
        // calculate errors
        //store predictions in an array
        for(int i = 0; i < OUTPUT_NODES; i++){
            if(layer2[i]>0.5){
                pred_arr[row] = 1;
            }
            else{
                pred_arr[row] = 0;
            }

        }
    }

    return 0;
}

float train_nn(float **input_matrix,
              float label[TRAIN_ROW],
              float **weight1,
              float **weight2,
              float layer1[HIDDEN_NODES],
              float layer2[OUTPUT_NODES])
              {
    //this will be each extracted input row
    float input[COL];

    for(int epoch=0; epoch < 100; epoch++){
      // iterate through input matrix row by row, extracting each row for training
      for(int row = 0; row < TRAIN_ROW; row++){
          for(int col = 0; col < COL; col++){
              input[col] = input_matrix[row][col];
          }
          //printf("%s\n", "hey");
          //this is for one row instance of forward and backprop
          // FORWARD PROPAGATION:
          for(int i = 0; i < HIDDEN_NODES; i++){
              float act = 0.0;
              for(int j = 0; j < INPUT_NODES; j++){
                  act += input[j]*weight1[i][j];
                  //printf("%f\n", act);
              }
              layer1[i] = sigmoid(act);
          }
          for(int i = 0; i < OUTPUT_NODES; i++){
              float act = 0.0;
              for(int j = 0; j < HIDDEN_NODES; j++){
                  act += layer1[j]*weight2[i][j];
              }
              layer2[i] = sigmoid(act);
          }
          //printf("layer 2: %f\n", layer2[0]);
          // BACKPROPAGATION:
          // calculate errors
          float d3[OUTPUT_NODES];
          for(int i = 0; i < OUTPUT_NODES; i++){
              float error_output = layer2[i] - label[row];
              d3[i] = error_output;
              //printf("%f\n", d3[i]);
          }
          //printf("%f\n", d3[0]);
          float d2[HIDDEN_NODES];
          for(int i = 0; i < HIDDEN_NODES; i++){
              float error_hidden = 0.0;
              for(int j = 0; j < OUTPUT_NODES; j++){
                  //printf("weight 2%f\n", weight2[j][i]);
                  //printf("d3 %f\n", d3[i]);
                  error_hidden += d3[j]*weight2[j][i];
                  //printf("error hidden%f\n", error_hidden);
              }
              d2[i] = error_hidden * (layer1[i] * (1 - layer1[i]));
              //printf("%f\n", d2[i]);
          }

          // update weights
          for(int i = 0; i < OUTPUT_NODES; i++){
              for(int j = 0; j < HIDDEN_NODES; j++){
                  weight2[i][j] -= layer1[j]*d3[i]*ALPHA;
              }
          }
          for(int i = 0; i < HIDDEN_NODES; i++){
              for(int j = 0; j < INPUT_NODES; j++){
                  weight1[i][j] -= input[j]*d2[i]*ALPHA;
              }
          }
      }
    }

    return 0;
}

int main(int argc, char *argv[]){
  //IMPORT TRAINING DATA
  float train_arr[TRAIN_ROW*COL];
  float *train_mat[TRAIN_ROW];
  for(int i = 0; i < TRAIN_ROW; i++){
      train_mat[i] = (float*)malloc(sizeof(COL*sizeof(float)));
  }
  FILE* str = fopen("train_data.csv", "r");

  char line[1024];
  int count = 0;
  while (fgets(line, 1024, str))
  {
    char* tmp = strdup(line);
    char* c = strtok(tmp,",");

    while(c != NULL){
      train_arr[count] = (float)atof(c);
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
  float *weight_layer1[HIDDEN_NODES];
  for(int i = 0; i < HIDDEN_NODES; i++){
      weight_layer1[i] = (float *)malloc(sizeof(INPUT_NODES*sizeof(float)));
  }

  float *weight_layer2[OUTPUT_NODES];
  for(int i = 0; i < OUTPUT_NODES; i++){
      weight_layer2[i] = (float *)malloc(sizeof(HIDDEN_NODES*sizeof(float)));
  }

  float *output = (float *)malloc(sizeof(TRAIN_ROW*sizeof(float)));

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

  float a = train_nn(train_mat, train_y_mat, weight_layer1, weight_layer2, layer1, layer2);

 float b = 0.0;
 b = predict(train_mat, output, weight_layer1, weight_layer2, layer1, layer2);

  printf("%s\n", "Weight layer 1 after training:");
  for(int i = 0; i < HIDDEN_NODES; i++){
      for(int j = 0; j < INPUT_NODES; j++){
          printf("%f ", weight_layer1[i][j]);
      }
      printf("\n");
  }
  printf("%s\n", "Weight layer 2 after training:");
  for(int i = 0; i < OUTPUT_NODES; i++){
      for(int j = 0; j < HIDDEN_NODES; j++){
          printf("%f ", weight_layer2[i][j]);
      }
      printf("\n");
  }

  int count_final=0;
  for(int i = 0; i < TRAIN_ROW; i++){
    printf("predicted %f\n", output[i]);
    printf("actual %f\n", train_y_arr[i]);
    if(output[i] == train_y_mat[i][0]){
        count_final +=1;
    }
  }

  printf("%d\n", count_final);

  return 0;
}
