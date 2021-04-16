#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define NX 8192
#define NY 8192

/* in microseconds (us) */
double get_elapsed_time(struct timeval *begin, struct timeval *end)
{
    return (end->tv_sec - begin->tv_sec) * 1000000
            + (end->tv_usec - begin->tv_usec);
}

void init(int rank, float *mdata, int N)
{
  int x, y;
  int cx = NX/2, cy = 0; /* center of ink */
  int rad = (NX+NY)/8; /* radius of ink */

  for(y = 0; y < N; y++) {
    int ny = rank*N+y;
    if (ny >= NY)
      break ;
    for(x = 0; x < NX; x++) {
      float v = 0.0;
      if (((x-cx)*(x-cx)+(ny-cy)*(ny-cy)) < rad*rad) {
	v = 1.0;
      }
      mdata[0*N*NX+y*NX+x] = v;
      mdata[1*N*NX+y*NX+x] = v;
    }
  }
  return;
}

/* Calculate for one time step */
/* Input: data[t%2], Output: data[(t+1)%2] */
void calc(int nt, int rank, int size, float *mdata, float *udata, float *bdata, int N)
{
  int t, x, y;
  MPI_Request ureq, breq;

  for (t = 0; t < nt; t++) {
    int from = t%2;
    int to = (t+1)%2;

    if (rank == 0) {
      printf("step %d\n", t);
      fflush(0);
    }

    if (rank > 0) {
      MPI_Status stat;
      MPI_Irecv((void *)udata, NX, MPI_FLOAT, rank-1, 100, MPI_COMM_WORLD, &ureq);
    }

    if (rank < size-1) {
      MPI_Status stat;
      MPI_Irecv((void *)bdata, NX, MPI_FLOAT, rank+1, 100, MPI_COMM_WORLD, &breq);
    }

    if (rank < size-1)
      for (y=1; y<N-1; y++)
	for (x=1; x<NX-1; x++)
	  mdata[to*N*NX+y*NX+x] = 0.2*(mdata[from*N*NX+y*NX+x]
				       + mdata[from*N*NX+y*NX+x-1]
				       + mdata[from*N*NX+y*NX+x+1]
				       + mdata[from*N*NX+(y-1)*NX+x]
				       + mdata[from*N*NX+(y+1)*NX+x]);

    if (rank == size-1) {
      int m = NY-rank*N;
      for (y=1; y<m-1; y++)
	for (x=1; x<NX-1; x++)
	  mdata[to*N*NX+y*NX+x] = 0.2*(mdata[from*N*NX+y*NX+x]
				       + mdata[from*N*NX+y*NX+x-1]
				       + mdata[from*N*NX+y*NX+x+1]
				       + mdata[from*N*NX+(y-1)*NX+x]
				       + mdata[from*N*NX+(y+1)*NX+x]);
    }

    if (rank > 0)
      MPI_Send((void *)(mdata+from*N*NX), NX, MPI_FLOAT, rank-1, 100, MPI_COMM_WORLD);

    if (rank < size-1)
      MPI_Send((void *)(mdata+from*N*NX+(N-1)*NX), NX, MPI_FLOAT, rank+1, 100, MPI_COMM_WORLD);

    if (rank > 0) {
      MPI_Status stat;
      MPI_Wait(&ureq, &stat);
    }

    if (rank < size-1) {
      MPI_Status stat;
      MPI_Wait(&breq, &stat);
    }

    if (rank > 0)
      for (x=1; x<NX-1; x++)
	mdata[to*N*NX+x] = 0.2*(mdata[from*N*NX+x]
				+ mdata[from*N*NX+x-1]
				+ mdata[from*N*NX+x+1]
				+ mdata[from*N*NX+NX+x]
				+ udata[x]);

    if (rank < size-1)
      for (x=1; x<NX-1; x++)
	mdata[to*N*NX+(N-1)*NX+x] = 0.2*(mdata[from*N*NX+(N-1)*NX+x]
					 + mdata[from*N*NX+(N-1)*NX+x-1]
					 + mdata[from*N*NX+(N-1)*NX+x+1]
					 + mdata[from*N*NX+(N-2)*NX+x]
					 + bdata[x]);
  }

  return;
}

int  main(int argc, char *argv[])
{
  struct timeval t1, t2;
  int nt = 20; /* number of time steps */
  
  if (argc >= 2) { /* if an argument is specified */
      nt = atoi(argv[1]);
  }

  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = (NY+size-1)/size;
  float mdata[2*N*NX], udata[NX], bdata[NX];

  init(rank, mdata, N);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
    gettimeofday(&t1, NULL);

  calc(nt, rank, size, mdata, udata, bdata, N);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    gettimeofday(&t2, NULL);

      double us;
      double gflops;
      int op_per_point = 5; // 4 add & 1 multiply per point

      us = get_elapsed_time(&t1, &t2);
      printf("Elapsed time: %.3lf sec\n", us/1000000.0);
      gflops = ((double)NX*NY*nt*op_per_point)/us/1000.0;
      printf("Speed: %.3lf GFlops\n", gflops);
  }

  MPI_Finalize();

  return 0;
}
