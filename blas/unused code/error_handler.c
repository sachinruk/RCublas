#include <stdio.h>
#include <stdlib.h>
#include "error_handler.h"

//prints out error message and exits program
void user_exit (const char *error_msg){
	int temp=0;
	printf(error_msg);
	scanf("%d",temp);/*wait till user reads error message*/
	exit(EXIT_FAILURE);
}

