#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NX 8192
#define NY 8192
#define N NX*NY

#define BS 1024

float data[N*2];

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
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;

  /* boundary condition*/ 
  if (i<NX) return ;
  if (i>=N-NX) return ;
  if (i%NX == 0) return ;
  if ((i+1)%NX == 0) return ;

  kdata[N*to+i] = 0.2 * (kdata[N*from+i]
			 + kdata[N*from+i-1]
			 + kdata[N*from+i+1]
			 + kdata[N*from+i-NX]
			 + kdata[N+from+i+NX]);
  return ;
}

/* Calculate for one time step */
/* Input: data[t%2], Output: data[(t+1)%2] */
void calc(int nt)
{
  int t;
  int *kdata;
  struct timeval st, st2, et, et2;
  double us, us2, flop;
  flop = (double)5.0*N*nt;

  gettimeofday(&st, NULL);

  cudaMalloc(&kdata, sizeof(float)*N*2);
  cudaMemcpy(kdata, data, sizeof(float)*N*2, cudaMemcpyDefault);
  cudaDeviceSynchronize();

  gettimeofday(&st2, NULL);

  for (t = 0; t < nt; t++) {
    int from = t%2;
    int to = (t+1)%2;

#if 1
    printf("step %d\n", t);
    fflush(0);
#endif

    calc_kernel<<<(N+BS-1)/BS, BS>>>(kdata, from, to);
//    cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();

  gettimeofday(&et2, NULL);

  cudaMemcpy(data, kdata, sizeof(float)*N*2, cudaMemcpyDefault);
  cudaDeviceSynchronize();

  gettimeofday(&et, NULL);

  us = get_elapsed_time(&st, &et);
  us2 = get_elapsed_time(&st2, &et2);

  printf("Calc took %.3lf us --> %.3lf Gflops (with data tranfer)\n", us, flop/us/1000.0);
  printf("Calc took %.3lf us --> %.3lf Gflops (withput data tranfer)\n", us2, flop/us2/1000.0);

  return ;
}

int  main(int argc, char *argv[])
{
  struct timeval t1, t2;
  int nt = 20; /* number of time steps */
  
  if (argc >= 2) { /* if an argument is specified */
      nt = atoi(argv[1]);
  }

  init();

  gettimeofday(&t1, NULL);

  calc(nt);

  gettimeofday(&t2, NULL);

  {
      double us;
      double gflops;
      int op_per_point = 5; // 4 add & 1 multiply per point

      us = get_elapsed_time(&t1, &t2);
      printf("Elapsed time: %.3lf sec\n", us/1000000.0);
      gflops = ((double)NX*NY*nt*op_per_point)/us/1000.0;
      printf("Speed: %.3lf GFlops\n", gflops);
  }

  return 0;
}
