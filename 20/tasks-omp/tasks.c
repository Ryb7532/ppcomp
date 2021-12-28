#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

int main(int argc, char *argv[])
{
#pragma omp parallel
#pragma omp single
  {
#pragma omp task
    {
      printf("I'm %d-th thread out of %d threads: Start A\n",
	     omp_get_thread_num(), omp_get_num_threads());
      sleep(4);
      printf("I'm %d-th thread out of %d threads: End A\n",
	     omp_get_thread_num(), omp_get_num_threads());
    }
    
#pragma omp task
    {
      printf("I'm %d-th thread out of %d threads: Start B\n",
	     omp_get_thread_num(), omp_get_num_threads());
      sleep(2);
      printf("I'm %d-th thread out of %d threads: End B\n",
	     omp_get_thread_num(), omp_get_num_threads());
    }
    
    printf("I'm %d-th thread out of %d threads: Start C\n",
	   omp_get_thread_num(), omp_get_num_threads());
    sleep(3);
    printf("I'm %d-th thread out of %d threads: End C\n",
	   omp_get_thread_num(), omp_get_num_threads());
    
#pragma omp taskwait
    
    printf("I'm %d-th thread out of %d threads: taskwait ended\n",
	   omp_get_thread_num(), omp_get_num_threads());
  } // parallel region ends

  return 0;
}
