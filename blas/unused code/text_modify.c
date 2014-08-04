#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "user_interface.h"
#include "error_handler.h"
#include "text_modify.h"
#include "mem.h"


#define FAIL			0
#define SUCCESS			1
#define INVALID_NUM		-1
#define CR				0x0D
#define LF				0x0A
#define MAX_NUM_LENGTH	33

#define NEW_LINE		printf("\n")

//private function (file scope)
FILE* file_open(char []);
void file_into_buf (FILE*, char **, int *);


void read_modify_txt (char name[]){
	char* counter_buf=NULL;
	int size_counter =0;
	unsigned long counter=0;
	FILE * file=NULL;
	int i=0;
	
	file=file_open(name);

	if (fseek(file,0,SEEK_SET))/*if file pointer is elsewhere make sure it is at beginning of text file*/
		user_exit("Could not rewind to original position");
	file_into_buf(file, &counter_buf,&size_counter);
	counter=atol(counter_buf);/*make string into a long int*/
	++counter;
	ltoa(counter, counter_buf,10);/*make long int back into a string*/
	if (fseek(file,0,SEEK_SET))/*if file pointer is elsewhere make sure it is at beginning of text file*/
		user_exit("Could not rewind to original position");/*exit*/
	fputs(counter_buf, file);/*so that it could be written to the beginning of file*/
	++i;/*increment i*/

	free (counter_buf);/*free pointer back into heap*/
	fclose(file);/*close file*/
}

//wrapper for fopen and checks for errors
//WARNING: exits program if file not found
FILE* file_open(char name[]){
	FILE* file=NULL;
	
	if ((file=fopen(name,"r+"))==NULL){/*if file does not exist */
		if ((file=fopen(name,"w"))==NULL)/*create it*/
			user_exit("could not create file");

		/*********************************************/
		/*This section may not be necassary depending on application*/
		if(fputs("0",file)<0)/*write a zero into it and if failed*/
			user_exit("could not write to stream");/*exit*/
		/********************************************/

		fclose(file);/*close file and*/
		if ((file=fopen(name,"r+"))==NULL)/*reopen in read and write mode*/
			user_exit("could not open file");
	}

	return file;
}
//puts contents of file into a buffer
void file_into_buf (FILE* file, char** buf, int* size){
	int i=0;

	while(1){
		if(i>=((*size)-1)){/*if number of characters gets bigger than the buffer*/
			(*size)=(*size)+10;/*increase capacity of buffer*/
			*buf=(char*)mem_alloc(*buf,(*size)*sizeof(char));/*allocate memory in blocks of 10*/
		}
		if(((*buf)[i]=(char)fgetc(file))==EOF){/*get characters from file AND if end of file reached*/
			break;/*break loop*/
		}
		++i;/*increment i*/
	}
	(*buf)[i]='\0';/*append null character*/
}


