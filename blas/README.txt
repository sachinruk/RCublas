blas00.c is a dummyfile
dllmain.c and prop.c are GPU initialisation code
fortran.c is the main interface between R or lapack between the Blas functions. Most likely place for bugs.
truncation.c is for use by single precision cards to truncate doubles to floats
print_vec.c is a debugging tool so that you can print out matrices, vectors onto text files