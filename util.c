#include "util.h"
#include <stdio.h>
#include <stdlib.h>

void raise_error(error_t error_type, const char *msg) {
  fprintf(stderr, "Error: %s\n", msg);
  exit(error_type);
}

