#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int m;
int n;
int k;
double *A;
double *B;
double *C;

long time_diff_us(struct timeval st, struct timeval et)
{
    return (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);
}

int matmulJLI()
{
    int i, j, l;
    int lda = m;
    int ldb = k;
    int ldc = m;
    struct timeval t0, t1, t2, t3;

    gettimeofday(&t0, NULL);
#pragma acc enter data copyin(A[0:m*k],B[0:k*n],C[0:m*n])
    gettimeofday(&t1, NULL);

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

    gettimeofday(&t2, NULL);
#pragma acc exit data copyout(C[0:m*n])
    gettimeofday(&t3, NULL);

    printf("\n");
    printf("%ld us for copyin, %ld us for computation, %ld us for copyout\n",
	   time_diff_us(t0, t1), time_diff_us(t1, t2), time_diff_us(t2, t3));

    return 0;
}

int matmulJIL()
{
    long i, j, l;
    long lda = m;
    long ldb = k;
    long ldc = m;
    struct timeval t0, t1, t2, t3;

    gettimeofday(&t0, NULL);
#pragma acc enter data copyin(A[0:m*k],B[0:k*n],C[0:m*n])
    gettimeofday(&t1, NULL);

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

    gettimeofday(&t2, NULL);
#pragma acc exit data copyout(C[0:m*n])
    gettimeofday(&t3, NULL);

    printf("\n");
    printf("%ld us for copyin, %ld us for computation, %ld us for copyout\n",
	   time_diff_us(t0, t1), time_diff_us(t1, t2), time_diff_us(t2, t3));

    return 0;
}

int main(int argc, char *argv[])
{
    int i, j;

    if (argc < 4) {
	printf("Specify M, N, K.\n");
	exit(1);
    }

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);

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
