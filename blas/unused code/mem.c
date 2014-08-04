#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include "mem.h"
#include "error_handler.h"

//wrapper for realloc and checks errors
void * mem_alloc (void* mem_block, size_t newsize){
	void* temp=NULL;
	/*printf("mem block %p, size %d\n", mem_block, newsize); */
	if((temp=realloc(mem_block,newsize))==NULL)
		user_exit("Memory allocation failure");
	//printf("new mem block %p\n", temp); 
	return temp;
}

