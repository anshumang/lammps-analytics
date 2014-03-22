/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "mpi.h"
#include "lammps.h"
#include "input.h"
#include "string.h"

#include <sys/time.h>
#include "cupti.h"

#include "analytics_kmeans_cuda_cu.h"

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
     const char *errstr;                                               \
     cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "rank %d : %s:%d: error: function %s failed with error %s.\n", \
              rank, __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))\
) : (buffer))


using namespace LAMMPS_NS;

int rank;
unsigned long cuptiStart, cuptiEnd; 
struct timeval cleStart, cleEnd;

static void printActivity(CUpti_Activity *record)
{
    if((record->kind == CUPTI_ACTIVITY_KIND_KERNEL) || (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)){
	CUpti_ActivityKernel2 *kernel = (CUpti_ActivityKernel2 *) record;
	//if (rank == 0){
	if (rank < 2){ //print for one simulation and one analytics process
		unsigned long long kernelStart = (cleStart.tv_sec * 1000000) + cleStart.tv_usec + ((kernel->start - cuptiStart)/1000);
		unsigned long long kernelSpan = ((unsigned long long) kernel->end - (unsigned long long) kernel->start)/1000;
		fprintf(stderr, "%s %llu %llu\n", kernel->name, kernelStart, kernelSpan);
	}
    }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE); 
    if (bfr == NULL) {
        fprintf(stderr, "rank %d : Error: out of memory\n", rank);
        exit(-1);
    }

    *size = BUF_SIZE;
    *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t  validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;

   if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                printActivity(record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else {
                CUPTI_CALL(status);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            printf("rank %d : Dropped %u activity records\n", rank, (unsigned int) dropped);
        }
    }

    free(buffer);
}
/* ----------------------------------------------------------------------
   main program to drive LAMMPS
------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
  MPI_Init(&argc,&argv);

  if(MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &rank))
	  fprintf(stderr, "MPI_Comm_rank failed\n");  

  MPI_Comm MPI_COMM_WORLD_SIM;

  int sim_rank;

  fprintf(stderr, "rank %d before MPI_Comm_split\n", rank);

  if(MPI_SUCCESS != MPI_Comm_split(MPI_COMM_WORLD,rank%2,rank,&MPI_COMM_WORLD_SIM))
	  fprintf(stderr, "MPI_Comm_split failed\n");

  if(MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD_SIM, &sim_rank))
	  fprintf(stderr, "MPI_Comm_rank failed\n");
  fprintf(stderr, "rank %d sim rank %d after MPI_Comm_split\n", rank, sim_rank);
  
  //if(rank%2 == 0){

	  fprintf(stderr, "rank %d sim rank %d in LAMMPS block\n", rank, sim_rank);

	  size_t attrValue = 0, attrValueSize = sizeof(size_t);

	  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
	  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

	  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
	  if(rank == 0)
		  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);

	  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
	  if(rank == 0)
		  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);

	  CUPTI_CALL(cuptiGetTimestamp(&cuptiStart));
	  gettimeofday(&cleStart, NULL);
	  fprintf(stderr, "rank %d : CUPTI start (ns) : %llu CLE start (us) : %llu\n", rank, cuptiStart, cleStart.tv_sec * 1000000 + cleStart.tv_usec);

  if(rank%2 == 0){
	  //LAMMPS *lammps = new LAMMPS(argc,argv,MPI_COMM_WORLD);
	  LAMMPS *lammps = new LAMMPS(argc,argv,MPI_COMM_WORLD_SIM);

	  lammps->input->file();
	  delete lammps;

	  //MPI_Barrier(MPI_COMM_WORLD);
	  fprintf(stderr, "LAMMPS rank %d waiting for others\n", rank);
	  MPI_Barrier(MPI_COMM_WORLD_SIM);
	  fprintf(stderr, "LAMMPS rank %d done waiting\n", rank);
	  
	  CUPTI_CALL(cuptiGetTimestamp(&cuptiEnd));
	  gettimeofday(&cleEnd, NULL);
	  fprintf(stderr, "rank %d : CUPTI end (ns) : %llu CLE end (us) : %llu\n", rank, cuptiEnd, cleEnd.tv_sec * 1000000 + cleEnd.tv_usec);

	  fprintf(stderr, "rank %d : CUPTI elapsed (ns) : %llu CLE elapsed (us) : %llu\n", rank, (cuptiEnd - cuptiStart), ((cleEnd.tv_sec - cleStart.tv_sec) * 1000000 + (cleEnd.tv_usec - cleStart.tv_usec)));

  }
  else {
	  fprintf(stderr, "rank %d in non LAMMPS block\n", rank);
	  char* argv[4] = {"kmeans", "-o", "-i", "kdd_cup"};
	  kmeans_wrap_main(4, (char **)&argv);
	  
	  CUPTI_CALL(cuptiGetTimestamp(&cuptiEnd));
	  gettimeofday(&cleEnd, NULL);
	  fprintf(stderr, "rank %d : CUPTI end (ns) : %llu CLE end (us) : %llu\n", rank, cuptiEnd, cleEnd.tv_sec * 1000000 + cleEnd.tv_usec);
	  fprintf(stderr, "rank %d : CUPTI elapsed (ns) : %llu CLE elapsed (us) : %llu\n", rank, (cuptiEnd - cuptiStart), ((cleEnd.tv_sec - cleStart.tv_sec) * 1000000 + (cleEnd.tv_usec - cleStart.tv_usec)));

  }

  fprintf(stderr, "LAMMPS+Analytics rank %d waiting for others\n", rank);
  MPI_Barrier(MPI_COMM_WORLD);
  fprintf(stderr, "LAMMPS+Analytics rank %d done waiting\n", rank);
  MPI_Finalize();
 
}
