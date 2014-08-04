#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mem.h"
#include "error_handler.h"
#include "user_interface.h"

//appends file name extension
void append_extension(char **file_name, const char * ext){
	size_t length_1=0, length_2=0;
	/*get length of strings*/
	length_1=strlen(ext);
	length_2=strlen(*file_name);	

	/*check whether extension already exists*/
	if ((strcmp(ext,*file_name+length_2-length_1))==0)
		return; /*no need to append anything*/
	else {
		*file_name=(char*)mem_alloc(*file_name,(length_1+1+length_2+1)*sizeof(char));
		strcat(*file_name,ext);/*concatenate extension to file name*/
	}
}

void get_file_name(char ** buffer){
	int i=0;

	printf("Input name of file <max length of file is 20 characters> and press ENTER\n");
	*buffer=mem_alloc(*buffer, 20*sizeof(char));
	fgets(*buffer,20,stdin);
	
	/*replace last \n with a \0*/
	for (i=0; i<20; i++){
		if((*buffer)[i]=='\n'){
			(*buffer)[i]='\0';
			i=REPLACED;
			break;
		}
	}
	
	if (i!=REPLACED)/*i acts as a flag indicating whether replacement took place*/
		user_exit("No new line character was encountered");

}

