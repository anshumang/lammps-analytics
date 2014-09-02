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

#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "analytics_kmeans_cuda_cu.h"
#include "analytics_histogram_cuda_cu.h"
#include "analytics_gaussian_cuda_cu.h"
#include "analytics_particlefilter_cuda_cu.h"
#include "analytics_sgemm_cuda_cu.h"
#include "analytics_stencil_cuda_cu.h"

#include "parboil_cutcp.h"
#include "parboil_histo/parboil_histo.h"
#include "parboil_stencil/parboil_stencil.h"

extern "C" int dummy_wrapper(int argc, char **argv, int rank, void *ptr);

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
int num_launch_cupti=0, num_memcpy_cupti=0;
FILE *cupti_tracking_sim_h, *cupti_tracking_ana_h;
//unsigned long start_timestamp[];

static void printActivityAna(CUpti_Activity *record)
{
    if(record->kind == CUPTI_ACTIVITY_KIND_KERNEL){
	CUpti_ActivityKernel2 *kernel = (CUpti_ActivityKernel2 *) record;
	if (rank == 0){ //print for one simulation and one analytics process
		unsigned long long kernelStart = (cleStart.tv_sec * 1000000) + cleStart.tv_usec + ((kernel->start - cuptiStart)/1000);
		unsigned long long kernelSpan = ((unsigned long long) kernel->end - (unsigned long long) kernel->start)/1000;
		//int launch_tag = kernel->blockX + kernel->blockY + kernel->blockZ + kernel->gridX + kernel->gridY + kernel->gridZ;
		num_launch_cupti++;
		//fprintf(stderr, "%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		//fprintf(cupti_tracking_h, "%llu\t%llu\t%s\t%u\n", kernelStart, kernelSpan, kernel->name, kernel->streamId);
	}
    }
    if(record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
	CUpti_ActivityKernel2 *kernel = (CUpti_ActivityKernel2 *) record;
	//if (rank == 0){ //print for one simulation and one analytics process
	if (rank == 1){ //print for one simulation and one analytics process
		unsigned long long kernelStart = (cleStart.tv_sec * 1000000) + cleStart.tv_usec + ((kernel->start - cuptiStart)/1000);
		unsigned long long kernelSpan = ((unsigned long long) kernel->end - (unsigned long long) kernel->start)/1000;
		//int launch_tag = kernel->blockX + kernel->blockY + kernel->blockZ + kernel->gridX + kernel->gridY + kernel->gridZ;
		num_launch_cupti++;
		if(rank == 0){
		//fprintf(stderr, "SIM\t%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		//fprintf(cupti_tracking_sim_h, "%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		}
		if(rank == 1){
		//fprintf(stderr, "ANA\t%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		fprintf(cupti_tracking_ana_h, "%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		}
		//fprintf(cupti_tracking_h, "%llu\t%llu\t%s\t%u\n", kernelStart, kernelSpan, kernel->name, kernel->streamId);
	}
    }
#if 0
    if(record->kind == CUPTI_ACTIVITY_KIND_MEMCPY){
        CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *) record;
        if (rank == 0){
                unsigned long long memcpyStart = (cleStart.tv_sec * 1000000) + cleStart.tv_usec + ((memcpy->start - cuptiStart)/1000);
                unsigned long long memcpySpan = ((unsigned long long) memcpy->end - (unsigned long long) memcpy->start)/1000;
                num_memcpy_cupti++;
                //fprintf(stderr, "M\t%llu\t%llu\n", memcpyStart, memcpySpan);
        }
    }
#endif
}

void CUPTIAPI bufferRequestedAna(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
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

void CUPTIAPI bufferCompletedAna(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t  validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;

   if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                printActivityAna(record);
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
static void printActivitySim(CUpti_Activity *record)
{
    if(record->kind == CUPTI_ACTIVITY_KIND_KERNEL){
	CUpti_ActivityKernel2 *kernel = (CUpti_ActivityKernel2 *) record;
	if (rank == 0){ //print for one simulation and one analytics process
		unsigned long long kernelStart = (cleStart.tv_sec * 1000000) + cleStart.tv_usec + ((kernel->start - cuptiStart)/1000);
		unsigned long long kernelSpan = ((unsigned long long) kernel->end - (unsigned long long) kernel->start)/1000;
		//int launch_tag = kernel->blockX + kernel->blockY + kernel->blockZ + kernel->gridX + kernel->gridY + kernel->gridZ;
		num_launch_cupti++;
		//fprintf(stderr, "%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		//fprintf(cupti_tracking_h, "%llu\t%llu\t%s\t%u\n", kernelStart, kernelSpan, kernel->name, kernel->streamId);
	}
    }
    if(record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
	CUpti_ActivityKernel2 *kernel = (CUpti_ActivityKernel2 *) record;
	if (rank == 0){ //print for one simulation and one analytics process
	//if (rank == 1){ //print for one simulation and one analytics process
		unsigned long long kernelStart = (cleStart.tv_sec * 1000000) + cleStart.tv_usec + ((kernel->start - cuptiStart)/1000);
		unsigned long long kernelSpan = ((unsigned long long) kernel->end - (unsigned long long) kernel->start)/1000;
		//int launch_tag = kernel->blockX + kernel->blockY + kernel->blockZ + kernel->gridX + kernel->gridY + kernel->gridZ;
		num_launch_cupti++;
		if(rank == 0){
		//fprintf(stderr, "SIM\t%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		fprintf(cupti_tracking_sim_h, "%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		}
		//if(rank == 1){
		//fprintf(stderr, "ANA\t%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		//fprintf(cupti_tracking_ana_h, "%llu\t%llu\t%s\n", kernelStart, kernelSpan, kernel->name);
		//}
		//fprintf(cupti_tracking_h, "%llu\t%llu\t%s\t%u\n", kernelStart, kernelSpan, kernel->name, kernel->streamId);
	}
    }
#if 0
    if(record->kind == CUPTI_ACTIVITY_KIND_MEMCPY){
        CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *) record;
        if (rank == 0){
                unsigned long long memcpyStart = (cleStart.tv_sec * 1000000) + cleStart.tv_usec + ((memcpy->start - cuptiStart)/1000);
                unsigned long long memcpySpan = ((unsigned long long) memcpy->end - (unsigned long long) memcpy->start)/1000;
                num_memcpy_cupti++;
                //fprintf(stderr, "M\t%llu\t%llu\n", memcpyStart, memcpySpan);
        }
    }
#endif
}

void CUPTIAPI bufferRequestedSim(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
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

void CUPTIAPI bufferCompletedSim(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t  validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;

   if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                printActivitySim(record);
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

//void *ptr_ana;
int main(int argc, char **argv)
{
  MPI_Init(&argc,&argv);

  if(MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD, &rank))
	  fprintf(stderr, "MPI_Comm_rank failed\n");  

  MPI_Comm MPI_COMM_WORLD_SIM_ANA;

  int sim_ana_rank;

  //fprintf(stderr, "%d : rank %d before MPI_Comm_split\n", getpid(), rank);

  if(MPI_SUCCESS != MPI_Comm_split(MPI_COMM_WORLD, rank%2, rank, &MPI_COMM_WORLD_SIM_ANA))
	  fprintf(stderr, "MPI_Comm_split failed\n");

  if(MPI_SUCCESS != MPI_Comm_rank(MPI_COMM_WORLD_SIM_ANA, &sim_ana_rank))
	  fprintf(stderr, "MPI_Comm_rank failed\n");

  //fprintf(stderr, "%d : rank %d sim/ana rank %d after MPI_Comm_split\n", getpid(), rank, sim_ana_rank);
 
  CUPTI_CALL(cuptiGetTimestamp(&cuptiStart));
  gettimeofday(&cleStart, NULL);
  //fprintf(stderr, "%d : rank %d : CUPTI start (ns) : %llu CLE start (us) : %llu\n", getpid(), rank, cuptiStart, cleStart.tv_sec * 1000000 + cleStart.tv_usec);

  if(rank%2 == 0){
	  //fprintf(stderr, "%d : rank %d sim rank %d in LAMMPS block\n", getpid(), rank, sim_ana_rank);
	  setenv("IS_ANALYTICS", "0", 1);
	  MPI_Barrier(MPI_COMM_WORLD_SIM_ANA);
	  size_t attrValue = 0, attrValueSize = sizeof(size_t);

	  cupti_tracking_sim_h = fopen("cupti_tracking_sim.txt", "w");
	  if(!cupti_tracking_sim_h){
		  fprintf(stderr, "Error opening file for dumping cupti trace for SIM\n");
	  }
	  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
	  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
	  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
	  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequestedSim, bufferCompletedSim));

	  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
	  if(rank == 0){
		  fprintf(stderr, "%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
	  }
	  attrValue = attrValue*2;
	  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
	  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
	  if(rank == 0){
		  fprintf(stderr, "%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
	  }

	  LAMMPS *lammps = new LAMMPS(argc,argv,MPI_COMM_WORLD_SIM_ANA);
	  //fprintf(stderr, "LAMMPS ctor called\n");

	  lammps->input->file();
	  //fprintf(stderr, "LAMMPS input file called\n");
	  delete lammps;

	  CUPTI_CALL(cuptiGetTimestamp(&cuptiEnd));
	  gettimeofday(&cleEnd, NULL);

	  //fprintf(stderr, "%d : rank %d : CUPTI end (ns) : %llu CLE end (us) : %llu\n", getpid(), rank, cuptiEnd, cleEnd.tv_sec * 1000000 + cleEnd.tv_usec);

          if(rank == 0){
		  fprintf(stderr, "SIM rank %d : CUPTI elapsed (ns) : %llu CLE elapsed (us) : %llu\n", rank, (cuptiEnd - cuptiStart), ((cleEnd.tv_sec - cleStart.tv_sec) * 1000000 + (cleEnd.tv_usec - cleStart.tv_usec)));
	  }
	  //fprintf(stderr, "%d : LAMMPS rank %d waiting for others\n", getpid(), sim_ana_rank);
	  MPI_Barrier(MPI_COMM_WORLD_SIM_ANA);
	  //fprintf(stderr, "%d : LAMMPS rank %d done waiting\n", getpid(), sim_ana_rank);
	  cuptiActivityFlushAll(0);
	  
  }
  else {
	  //fprintf(stderr, "%d : rank %d ana rank %d in non LAMMPS block\n", getpid(), rank, sim_ana_rank);
	  setenv("IS_ANALYTICS", "1", 1);
	  MPI_Barrier(MPI_COMM_WORLD_SIM_ANA);
	  size_t attrValue = 0, attrValueSize = sizeof(size_t);

	  cupti_tracking_ana_h = fopen("cupti_tracking_ana.txt", "w");
	  if(!cupti_tracking_ana_h){
		  fprintf(stderr, "Error opening file for dumping cupti trace for ANA\n");
	  }
	  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
	  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
	  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
	  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequestedAna, bufferCompletedAna));

	  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
	  if(rank == 0){
		  fprintf(stderr, "%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
	  }
	  //attrValue = attrValue*8;
	  //CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
	  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
	  if(rank == 0){
		  fprintf(stderr, "%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
	  }

	  //char* argv[4] = {"kmeans", "-o", "-i", "kdd_cup"};
	  //kmeans_wrap_main(4, (char **)&argv);

	  //char *argv[1] = {"histogram"};
	  //histogram_wrap_main(1, (char **)&argv);

	  //char* argv[] = {"gaussian", "-s", "32"};
	  //gaussian_wrap_main(3, (char **)argv);

	  //char *argv[] = {"particle_filter", "-x", "128", "-y", "128", "-z", "10", "-np", "1000"};
	  //particlefilter_wrap_main(9, (char **)argv);

	  //char *argv[7] = {"sgemm", "-o", "sgemm_out.txt", "-i", "matrix1.txt,matrix2.txt,matrix2t.txt","--"};
	  //sgemm_wrap_main(7, (char **)&argv);

	  //char *argv[8] = {"stencil", "128", "128", "32", "100", "-i", "128x128x32.bin", "--"};
	  //stencil_wrap_main(8, (char **)argv, sim_ana_rank);

	  //char *argv[] = {"parboil_cutcp", "-i", "watbox.sl100.pqr"};
	  //parboil_cutcp_wrapper(3, (char **)argv, sim_ana_rank);
	 
	  void *ptr = NULL; 
	  //char *argv[] = {"parboil_histo", "10000", "-i", "img.bin"}; //10
	  //char *argv[] = {"parboil_histo", "20000", "-i", "img.bin"}; //20
	  //char *argv[] = {"parboil_histo", "30000", "-i", "img.bin"}; //30
	  //char *argv[] = {"parboil_histo", "40000", "-i", "img.bin"}; //40
	  char *argv[] = {"parboil_histo", "50000", "-i", "img.bin"}; //50
	  parboil_histo_wrapper(4, (char **)argv, sim_ana_rank, ptr);

	  //char *argv[] = {"parboil_stencil", "512", "512", "64", "100", "-i", "512x512x64x100.bin"};
	  //parboil_stencil_wrapper(7, (char **)argv, sim_ana_rank, ptr);

	  CUPTI_CALL(cuptiGetTimestamp(&cuptiEnd));
	  gettimeofday(&cleEnd, NULL);
	  //fprintf(stderr, "%d : rank %d : CUPTI end (ns) : %llu CLE end (us) : %llu\n", getpid(), rank, cuptiEnd, cleEnd.tv_sec * 1000000 + cleEnd.tv_usec);
          if(rank == 1){
		  fprintf(stderr, "ANA rank %d : CUPTI elapsed (ns) : %llu CLE elapsed (us) : %llu\n", rank, (cuptiEnd - cuptiStart), ((cleEnd.tv_sec - cleStart.tv_sec) * 1000000 + (cleEnd.tv_usec - cleStart.tv_usec)));
	  }

	  //fprintf(stderr, "%d : Analytics rank %d waiting for others\n", getpid(), sim_ana_rank);
	  MPI_Barrier(MPI_COMM_WORLD_SIM_ANA);
	  cuptiActivityFlushAll(0);
	  //fprintf(stderr, "%d : Analytics rank %d done waiting\n", getpid(), sim_ana_rank);
  }

  //fprintf(stderr, "%d : LAMMPS+Analytics rank %d waiting for others\n", getpid(), rank);
  MPI_Barrier(MPI_COMM_WORLD);
  //int ret = fclose(cupti_tracking_ana_h);
  //if(ret != 0){
    //fprintf("Error closing cupti_tracking_ana.txt\n");
  //}
  //fprintf(stderr, "%d : LAMMPS+Analytics rank %d done waiting\n", getpid(), rank);
  MPI_Finalize();
  //fprintf(stderr, "CUPTI kernel = %d\tmemcpy = %d\n", num_launch_cupti, num_memcpy_cupti);
 
}
