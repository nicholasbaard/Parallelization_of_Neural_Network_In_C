#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROW 773
#define COL 27

int main(int argc, char *argv[]){
  float arr[ROW*COL];
  FILE* str = fopen("audit_risk_raw.csv", "r");

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

  for (size_t i = 0; i < ROW; i++) {
    for (size_t j = 0; j < COL; j++) {
      printf("%f ", arr[i*COL + j]);
    }
    printf("\n");
  }
  return 0;
}
