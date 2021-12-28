#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

long m;
long n;
long k;
double *A;
double *B;
double *C;

long time_diff_us(struct timeval st, struct timeval et)
{
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

int matmulJLI()
{
    long i, j, l;
    long lda = m;
    long ldb = k;
    long ldc = m;
#pragma acc data copyin(A[0:m*k],B[0:k*n]),copy(C[0:m*n])
#pragma acc kernels
#pragma acc loop independent
    for (j = 0; j < n; j++) {
#pragma acc loop seq
	for (l = 0; l < k; l++) {
	    double blj = B[l+j*ldb];
#pragma acc loop independent
	    for (i = 0; i < m; i++) {
		double ail = A[i+l*lda];
		C[i+j*ldc] += ail*blj;
	    }
	}
    }

    return 0;
}

int matmulJIL()
{
    long i, j, l;
    long lda = m;
    long ldb = k;
    long ldc = m;
#pragma acc data copyin(A[0:m*k],B[0:k*n]),copy(C[0:m*n])
#pragma acc kernels
#pragma acc loop independent
    for (j = 0; j < n; j++) {
#pragma acc loop independent
	for (i = 0; i < m; i++) {
	    double cij = C[i+j*ldc];
#pragma acc loop seq
	    for (l = 0; l < k; l++) {
		double blj = B[l+j*ldb];
		double ail = A[i+l*lda];
		cij += ail*blj;
	    }
	    C[i+j*ldc] = cij;
	}
    }
    return 0;
}

int main(int argc, char *argv[])
{
    long i, j;

    if (argc < 4) {
	printf("Specify M, N, K.\n");
	exit(1);
    }

    m = atol(argv[1]);
    n = atol(argv[2]);
    k = atol(argv[3]);

    /* allocate matrix region */
    A = (double *)malloc(sizeof(double)*m*k);
    B = (double *)malloc(sizeof(double)*k*n);
    C = (double *)malloc(sizeof(double)*m*n);

    /* setup matrix (column major) */
    /* A is m*k matrix */
    for (j = 0; j < k; j++) {
	for (i = 0; i < m; i++) {
	    A[i+j*m] = 1.0;
	}
    }
    /* B is k*n matrix */
    for (j = 0; j < n; j++) {
	for (i = 0; i < k; i++) {
	    B[i+j*k] = 10.0;
	}
    }
    /* C is m*n matrix */
    for (j = 0; j < n; j++) {
	for (i = 0; i < m; i++) {
	    C[i+j*m] = 0.0;
	}
    }

    /* Repeat same computation for 3 times */
    for (i = 0; i < 3; i++) {
	struct timeval st;
	struct timeval et;
	long us;

	gettimeofday(&st, NULL); /* get start time */
	matmulJIL();
	//matmulJLI();
	gettimeofday(&et, NULL); /* get start time */

	us = time_diff_us(st, et);
	printf("Matmul took %ld us --> %.3lf GFlops\n",
	       us, 2.0*(double)m*(double)n*(double)k/(double)us/1000.0);
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
