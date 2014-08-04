#include <stdio.h>
#include "print_vec.h"

void print_vector(void * vec, char type, int elements){
	unsigned int i=0;
	int tempI=0;
	float *tempF=NULL;
	double *tempD=NULL;
	FILE *file=NULL;

	file=fopen("debug.txt","a+");/*append to debug*/
	fprintf (file,"\nVECTOR of elements %d\n", elements);/*start of new vector/ matrix with 2 new lines*/
	if (type==FLOAT){/*if a single precision vector*/
		tempF=(float *)vec;/*copy to temporary pointer*/
		for (i=0;i<elements;i++){
			fprintf(file,"%f\t",tempF[i]);/*print values*/
			if ((i+1)%10==0)/*Every 10th element*/
				fprintf(file,"\n");/*print a new line*/
		}
	}
	else if (type==DOUBLE){/*if double precision vector*/
		tempD=(double *)vec;/*copy to temporary pointer*/
		for (i=0;i<elements;i++){
			fprintf(file,"%f\t",tempD[i]);/*print values*/
			if ((i+1)%10==0)/*Every 10th element*/
				fprintf(file,"\n");/*print a new line*/
		}
	}

	else if (type==INT){
		tempI=*((int *)vec);
		fprintf(file, "%d\n",tempI);
	}

	fclose(file);/*close file*/

}

void print_matrix(void * vec, size_t size,int rows, int elements){
	unsigned int i=0;
	float *tempF=NULL;
	double *tempD=NULL;
	FILE *file=NULL;

	file=fopen("debug.txt","a+");/*append to debug*/
	fprintf (file, "\nMATRIX of elements %d rows %d\n",elements, rows);/*start of new vector/ matrix with 2 new lines*/

	if (size==sizeof(float)){/*if a single precision vector*/
		tempF=(float *)vec;/*copy to temporary pointer*/
		for (i=0;i<elements;i++){
			fprintf(file,"%.2f\t",tempF[i]);/*print values*/
			if ((i+1)%rows==0)
				fprintf(file,"\n");
		}
	}
	else if (size==sizeof(double)){/*if double precision vector*/
		tempD=(double *)vec;/*copy to temporary pointer*/
		for (i=0;i<elements;i++){
			fprintf(file,"%.2E\t",tempD[i]);/*print values*/
			if ((i+1)%rows==0)
					fprintf(file,"\n");
		}
	}

	fclose (file);/*close file*/

}

void print_func_name (char *name){
	FILE *file=NULL;

	file=fopen("debug.txt","a+");/*append to debug*/
	fprintf (file, "\n\n%s\n",name);/*start of new vector/ matrix with 2 new lines*/
	fclose(file);
}

void print_error (char *msg){
	FILE *file=NULL;

	file=fopen("error_log.txt","a+");/*append to debug*/
	fprintf (file, "\n\n%s\n",msg);/*start of new vector/ matrix with 2 new lines*/
	fclose(file);
}



	


