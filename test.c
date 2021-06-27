#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


int main(int argc, char *argv[]){

  float matrix[4][3];

  int num_rows = sizeof(matrix) / sizeof(matrix[0]);
  int num_cols = sizeof(matrix[0]) / sizeof(matrix[0][0]);

  printf("num rows %d\n", num_rows);
  printf("num cols %d\n", num_cols);

  return 0;
}
