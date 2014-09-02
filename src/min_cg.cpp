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

#include "lmptype.h"
#include "mpi.h"
#include "math.h"
#include "string.h"
#include "min_cg.h"
#include "atom.h"
#include "update.h"
#include "output.h"
#include "timer.h"
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include "stdlib.h"
#include <mqueue.h>
#include <errno.h>

using namespace LAMMPS_NS;

// EPS_ENERGY = minimum normalization for energy tolerance

#define EPS_ENERGY 1.0e-8

// same as in other min classes

enum{MAXITER,MAXEVAL,ETOL,FTOL,DOWNHILL,ZEROALPHA,ZEROFORCE,ZEROQUAD};

/* ---------------------------------------------------------------------- */

MinCG::MinCG(LAMMPS *lmp) : MinLineSearch(lmp) {}

/* ----------------------------------------------------------------------
   minimization via conjugate gradient iterations
------------------------------------------------------------------------- */
//void *ptr=NULL;
int g_iter = 0, ack = 0, my_sim_g_rank = 0;
mqd_t mqd_sim;
//unsigned *iterFlag=NULL;
int MinCG::iterate(int maxiter)
{
  //fprintf(stderr, "Entering MinCG::iterate\n");
  int i,m,n,fail,ntimestep;
  double beta,gg,dot[2],dotall[2];
  double *fatom,*gatom,*hatom;

  // nlimit = max # of CG iterations before restarting
  // set to ndoftotal unless too big

  int nlimit = static_cast<int> (MIN(MAXSMALLINT,ndoftotal));

  // initialize working vectors

  for (i = 0; i < nvec; i++) h[i] = g[i] = fvec[i];
  if (nextra_atom)
    for (m = 0; m < nextra_atom; m++) {
      fatom = fextra_atom[m];
      gatom = gextra_atom[m];
      hatom = hextra_atom[m];
      n = extra_nlen[m];
      for (i = 0; i < n; i++) hatom[i] = gatom[i] = fatom[i];
    }
  if (nextra_global)
    for (i = 0; i < nextra_global; i++) hextra[i] = gextra[i] = fextra[i];

  gg = fnorm_sqr();

  int rank=-1;
  MPI_Comm_rank(world, &rank);
  //fprintf(stderr, "SIM rank %d\n", rank);

#if 0
  char *memname = "SIM_ANA_SYNC_SHM";
  /*if(0 != shm_unlink(memname)){
	  perror("shm_unlink SIM");
	  exit(EXIT_FAILURE);
  }
  fprintf(stderr, "SIM shm_unlink at start done\n");
  */
  int fd = shm_open(memname, O_CREAT | O_TRUNC | O_RDWR, 0666);
  if (fd == -1){
	  perror("shm_open SIM");
	  exit(EXIT_FAILURE);
  }
  fprintf(stderr, "SIM shm_open done fd %d\n", fd);

  size_t region_size = sysconf(_SC_PAGE_SIZE);
  fprintf(stderr, "SIM region_size=%d\n", region_size);

  if(ftruncate(fd, region_size)!=0){
	  perror("ftruncate SIM");
	  exit(EXIT_FAILURE);
  }
  fprintf(stderr, "SIM ftruncate done\n");

  void *ptr = mmap(0, region_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED){
	  perror("mmap SIM");
	  exit(EXIT_FAILURE);
  }
  fprintf(stderr, "SIM mmap done ptr %p\n", ptr);

  close(fd);
  fprintf(stderr, "SIM close done\n");
  unsigned *iterFlag = (unsigned *)ptr;
  //unsigned val;
  //iterFlag = &val;
  *iterFlag = 0;
#if 0
  for (int iter = 0; iter < 10; iter++) {
    *iterFlag = iter;
    if(rank==0){
	    fprintf(stderr, "---------SIM start iter---------- %d\n", *iterFlag);
    }
    while(*iterFlag != 0);
    if(rank==0){
	    fprintf(stderr, "---------SHM update seen by SIM----------\n");
    }
  }
#endif
#endif

  my_sim_g_rank = rank;
#if 1
  char *path = "/tmp";
  struct mq_attr buf;
  buf.mq_msgsize = sizeof(int);
  buf.mq_maxmsg = 10;
  if(-1 == mq_unlink (path)){
	  perror("mq_unlink()");
  }
  //mqd_t mqd = mq_open(path,O_RDWR|O_NONBLOCK|O_CREAT|O_EXCL,0666,&buf);
  /*mqd_t */mqd_sim = mq_open(path,O_RDWR|O_CREAT|O_EXCL,0666,&buf);
  if (-1 != mqd_sim){
      fprintf(stderr, "LAMMPS opening queue %d\n", mqd_sim);
  }else{
      perror("mq_open() LAMMPS");
  }
#endif

#if 1
  struct timeval start, curr, begin, end;
  gettimeofday(&begin, NULL);
  for (int iter = 0; iter < maxiter; iter++) {
  //for (int iter = 0; iter < 17; iter++) {
    unsigned long curr_time;
    gettimeofday(&start, NULL);
    ntimestep = ++update->ntimestep;
    niter++;
    //g_iter = iter;
        
    struct timeval curr;
    //unsigned long curr_time;
    gettimeofday(&curr, NULL);
    curr_time = curr.tv_sec * 1000000 + curr.tv_usec;
    //if(my_sim_g_rank == 0){
    //	  fprintf(stderr, "MIN\t%llu\t%d\n", curr_time, g_iter);
    //}
    //g_iter++;
    //*iterFlag = g_iter;
    //*((unsigned *)ptr) = iter;
    //if(rank==0){
    //	    fprintf(stderr, "---------SIM start iter---------- %d \t %d\n", g_iter, *iterFlag);
    //}
#if 0
    if (-1 == mq_send(mqd_sim, (const char *)&g_iter, sizeof(int), 0)){
	//perror("mq_send() LAMMPS sending new step");
    }else{
	    if(rank == 0){
		    fprintf(stderr, "LAMMPS sending new step %d\n", g_iter);
	    }
    }
#endif
    //while(*iterFlag != 0);
/*
    ack = 0;
    if (-1 == mq_receive(mqd, (char *)&ack, sizeof(int), 0)){
	//perror("mq_send() LAMMPS receiving ACK");
    }else{
	    if(rank == 0){
		    //fprintf(stderr, "LAMMPS received ack for new step %d %d\n", g_iter, ack);
	    }
    }
    if(ack != g_iter){
	if(rank == 0){
		fprintf(stderr, "Did not receive proper ack %d\n", ack);
	}
    }
*/
    //if(rank==0){
    //	    fprintf(stderr, "---------SIM start iter seen by ANA---------- %d\n", *iterFlag);
    //}

    // line minimization along direction h from current atom->x
#if 1
    eprevious = ecurrent;
    //gettimeofday(&curr, NULL);
    //curr_time = curr.tv_sec * 1000000 + curr.tv_usec;
    if(rank == 0){
       //fprintf(stderr, "Before linemin \t %ld\n", curr_time);
    } 
    fail = (this->*linemin)(ecurrent,alpha_final);
    g_iter++;
    if (fail) return fail;
    //gettimeofday(&curr, NULL);
    //curr_time = curr.tv_sec * 1000000 + curr.tv_usec;
    if(rank == 0){
       //fprintf(stderr, "After linemin \t %ld\n", curr_time);
    } 

    // function evaluation criterion

    if (neval >= update->max_eval) return MAXEVAL;

    // energy tolerance criterion

    if (fabs(ecurrent-eprevious) <
        update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
      return ETOL;

    // force tolerance criterion

    dot[0] = dot[1] = 0.0;
    for (i = 0; i < nvec; i++) {
      dot[0] += fvec[i]*fvec[i];
      dot[1] += fvec[i]*g[i];
    }
    if (nextra_atom)
      for (m = 0; m < nextra_atom; m++) {
        fatom = fextra_atom[m];
        gatom = gextra_atom[m];
        n = extra_nlen[m];
        for (i = 0; i < n; i++) {
          dot[0] += fatom[i]*fatom[i];
          dot[1] += fatom[i]*gatom[i];
        }
      }
    //gettimeofday(&curr, NULL);
    //curr_time = curr.tv_sec * 1000000 + curr.tv_usec;
    if(rank == 0){
       //fprintf(stderr, "Before Allreduce [1] \t %ld\n", curr_time);
    } 
    MPI_Allreduce(dot,dotall,2,MPI_DOUBLE,MPI_SUM,world);
    //curr_time = curr.tv_sec * 1000000 + curr.tv_usec;
    //gettimeofday(&curr, NULL); 
    if(rank == 0){
       //fprintf(stderr, "After Allreduce [1] \t %ld\n", curr_time);
    } 
    //curr_time = curr.tv_sec * 1000000 + curr.tv_usec;
    if (nextra_global)
      for (i = 0; i < nextra_global; i++) {
        dotall[0] += fextra[i]*fextra[i];
        dotall[1] += fextra[i]*gextra[i];
      }

    if (dotall[0] < update->ftol*update->ftol) return FTOL;

    // update new search direction h from new f = -Grad(x) and old g
    // this is Polak-Ribieri formulation
    // beta = dotall[0]/gg would be Fletcher-Reeves
    // reinitialize CG every ndof iterations by setting beta = 0.0

    beta = MAX(0.0,(dotall[0] - dotall[1])/gg);
    if ((niter+1) % nlimit == 0) beta = 0.0;
    gg = dotall[0];

    for (i = 0; i < nvec; i++) {
      g[i] = fvec[i];
      h[i] = g[i] + beta*h[i];
    }
    if (nextra_atom)
      for (m = 0; m < nextra_atom; m++) {
        fatom = fextra_atom[m];
        gatom = gextra_atom[m];
        hatom = hextra_atom[m];
        n = extra_nlen[m];
        for (i = 0; i < n; i++) {
          gatom[i] = fatom[i];
          hatom[i] = gatom[i] + beta*hatom[i];
        }
      }
    if (nextra_global)
      for (i = 0; i < nextra_global; i++) {
        gextra[i] = fextra[i];
        hextra[i] = gextra[i] + beta*hextra[i];
      }

    // reinitialize CG if new search direction h is not downhill

    dot[0] = 0.0;
    for (i = 0; i < nvec; i++) dot[0] += g[i]*h[i];
    if (nextra_atom)
      for (m = 0; m < nextra_atom; m++) {
        gatom = gextra_atom[m];
        hatom = hextra_atom[m];
        n = extra_nlen[m];
        for (i = 0; i < n; i++) dot[0] += gatom[i]*hatom[i];
      }
    //gettimeofday(&curr, NULL);
    //curr_time = curr.tv_sec * 1000000 + curr.tv_usec;
    if(rank == 0){
       //fprintf(stderr, "=================================Before Allreduce [2] \t %ld\n", curr_time);
    } 
    MPI_Allreduce(dot,dotall,1,MPI_DOUBLE,MPI_SUM,world);
    //gettimeofday(&curr, NULL);
    //curr_time = curr.tv_sec * 1000000 + curr.tv_usec;
    if(rank == 0){
       //fprintf(stderr, "=================================After Allreduce [2] \t %ld\n", curr_time);
    } 
    if (nextra_global)
      for (i = 0; i < nextra_global; i++)
        dotall[0] += gextra[i]*hextra[i];

    if (dotall[0] <= 0.0) {
      for (i = 0; i < nvec; i++) h[i] = g[i];
      if (nextra_atom)
        for (m = 0; m < nextra_atom; m++) {
          gatom = gextra_atom[m];
          hatom = hextra_atom[m];
          n = extra_nlen[m];
          for (i = 0; i < n; i++) hatom[i] = gatom[i];
        }
      if (nextra_global)
        for (i = 0; i < nextra_global; i++) hextra[i] = gextra[i];
    }

    // output for thermo, dump, restart files

    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(TIME_OUTPUT);
    }
    //gettimeofday(&curr, NULL);
    //unsigned long start_time = start.tv_sec * 1000000 + start.tv_usec;
    //curr_time = curr.tv_sec * 1000000 + curr.tv_usec;
    if(rank == 0){
	    //fprintf(stderr, "==========================================%d\t%ld\t%ld\t%ld\n", iter, (curr.tv_sec - start.tv_sec)*1000000 + (curr.tv_usec - start.tv_usec), start_time, curr_time);
	    //fprintf(stderr, "SIM timestep %d %ld\n", iter, (curr.tv_sec - start.tv_sec)*1000000 + (curr.tv_usec - start.tv_usec));
	    //fprintf(stderr, "========================================== SIM timestep (rank %d) %d\n", rank, iter);
    }
#endif
  }
  //gettimeofday(&end, NULL);
  //if(rank == 0){
	  //fprintf(stderr, " END (rank %d) %ld\n", rank, (end.tv_sec - begin.tv_sec)*1000000 + (end.tv_usec - begin.tv_usec));
  //}
#endif
#if 0
  if(0 != munmap(ptr, region_size)){
	  perror("munmap SIM");
	  exit(EXIT_FAILURE);
  }
  fprintf(stderr, "SIM munmap done\n");

  if(0 != shm_unlink(memname)){
	  perror("shm_unlink SIM");
	  exit(EXIT_FAILURE);
  }
  fprintf(stderr, "SIM shm_unlink done\n");
#endif
  return MAXITER;
}
