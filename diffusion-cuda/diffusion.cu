#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NX 8192
#define NY 8192
#define N NX*NY

#define BS 16

float data[N*2];
double sum = 0.0, sum1 = 0.0, sum2 = 0.0;

/* in microseconds (us) */
double get_elapsed_time(struct timeval *begin, struct timeval *end)
{
    return (end->tv_sec - begin->tv_sec) * 1000000
            + (end->tv_usec - begin->tv_usec);
}

void init()
{
  int x, y;
  int cx = NX/2, cy = 0; /* center of ink */
  int rad = (NX+NY)/8; /* radius of ink */

  for(y = 0; y < NY; y++) {
    for(x = 0; x < NX; x++) {
      float v = 0.0;
      if (((x-cx)*(x-cx)+(y-cy)*(y-cy)) < rad*rad) {
	v = 1.0;
      }
      data[N*0+NX*y+x] = v;
      data[N*1+NX*y+x] = v;
    }
  }
  return;
}

__global__ void calc_kernel(int *kdata, int from, int to) {
  int i, j;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;

  /* boundary condition*/ 
  if (i==0) return ;
  if (i>=NX-1) return ;
  if (j==0) return ;
  if (j>=NY-1) return ;

  kdata[N*to+NX*j+i] = 0.2 * (kdata[N*from+NX*j+i]
			 + kdata[N*from+NX*j+i-1]
			 + kdata[N*from+NX*j+i+1]
			 + kdata[N*from+NX*j+i-NX]
			 + kdata[N+from+NX*j+i+NX]);
  return ;
}

/* Calculate for one time step */
/* Input: data[t%2], Output: data[(t+1)%2] */
void calc(int nt)
{
  int t;
  int *kdata;

  struct timeval st, st2, et, et2;

  dim3 gridDim = dim3((NX+BS-1)/BS, (NY+BS-1)/BS, 1);
  dim3 blockDim = dim3(BS, BS, 1);

  gettimeofday(&st, NULL);

  cudaMalloc(&kdata, sizeof(float)*N*2);
  cudaMemcpy(kdata, data, sizeof(float)*N*2, cudaMemcpyDefault);
  cudaDeviceSynchronize();

  gettimeofday(&st2, NULL);

  for (t = 0; t < nt; t++) {
    int from = t%2;
    int to = (t+1)%2;

/*
#if 1
    printf("step %d\n", t);
    fflush(0);
#endif
*/

    calc_kernel<<<gridDim, blockDim>>>(kdata, from, to);
  }
  cudaDeviceSynchronize();

  gettimeofday(&et2, NULL);

  cudaMemcpy(data, kdata, sizeof(float)*N*2, cudaMemcpyDefault);
  cudaDeviceSynchronize();

  gettimeofday(&et, NULL);

  sum1 += get_elapsed_time(&st, &et);
  sum2 += get_elapsed_time(&st2, &et2);

  return ;
}

int  main(int argc, char *argv[])
{
  struct timeval t1, t2;
  int nt = 20, i; /* number of time steps */

  if (argc >= 2) { /* if an argument is specified */
      nt = atoi(argv[1]);
  }
  for (i=0; i<10; i++) {
  init();

  gettimeofday(&t1, NULL);

  calc(nt);

  gettimeofday(&t2, NULL);

  {
      double us;
      double gflops;
      int op_per_point = 5; // 4 add & 1 multiply per point

      us = get_elapsed_time(&t1, &t2);
      sum += us;
      printf("Elapsed time: %.3lf sec\n", us/1000000.0);
      gflops = ((double)NX*NY*nt*op_per_point)/us/1000.0;
      printf("Speed: %.3lf GFlops\n", gflops);
  }
  }


  sum /= 10; sum1 /= 10; sum2 /= 10;
  printf("average time: %.3lf sec\n", sum/1000000.0);
  printf("average speed: %.3lf GFlops\n", ((double)NX*NY*nt*5)/sum/1000.0);
  printf("average speed(with transfer): %.3lf GFlops\n", ((double)NX*NY*nt*5)/sum1/1000.0);
  printf("average speed(without transfer): %.3lf GFlops\n", ((double)NX*NY*nt*5)/sum2/1000.0);
  return 0;
}
