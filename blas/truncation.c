#include <stdlib.h>
#include <malloc.h>
#include "print_vec.h"
#include "truncation.h"

void untruncate (void * array_, unsigned int elements){
	float *temp=NULL;
	double *temp2=NULL;
	/*FILE *file=NULL;*/

	temp=(float *)array_;
	temp2=(double *)array_;

	/*file=fopen("errorlog.txt","a+");
	fprintf(file,"elements :%d",elements);*/
	
	while (elements>0){
		--elements;
		temp2[elements]=(double)temp[elements];
		/*fprintf(file,"%.2E\t",temp2[elements]);*/
	}
	/*fprintf(file,"\n");
	fclose(file);*/
}

void truncate (void *array_, unsigned int elements){
	float *temp=NULL;
	double *temp2=NULL;
	unsigned int i=0;

	temp=(float *)array_;
	temp2=(double *)array_;

	while (i<elements){
		temp[i]=(float)temp2[i];
		++i;
	}
}
//Copy to a new array while truncating
float * copy_n_truncate (const double *array_,int elements){
	int i=0;
	float *temp=NULL;
	temp=(float *)malloc((size_t)elements*sizeof(float));

	if (temp==NULL)
		print_error("Could not allocate sufficient memory");

	for (i=0;i<elements;i++)
		temp[i]=(float)array_[i];

	return temp;
}

void copy_untruncate (const float *src, double *dest, unsigned int elements){
	int i=0;

	for (i=0; i<elements; i++)
		dest[i]=(double)src[i];

}

