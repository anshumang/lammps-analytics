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
#include "string.h"
#include "ctype.h"
#include "lammps.h"
#include "style_angle.h"
#include "style_atom.h"
#include "style_bond.h"
#include "style_command.h"
#include "style_compute.h"
#include "style_dihedral.h"
#include "style_dump.h"
#include "style_fix.h"
#include "style_improper.h"
#include "style_integrate.h"
#include "style_kspace.h"
#include "style_minimize.h"
#include "style_pair.h"
#include "style_region.h"
#include "universe.h"
#include "input.h"
#include "atom.h"
#include "update.h"
#include "neighbor.h"
#include "comm.h"
#include "domain.h"
#include "force.h"
#include "modify.h"
#include "group.h"
#include "output.h"
#include "citeme.h"
#include "accelerator_cuda.h"
#include "accelerator_kokkos.h"
#include "accelerator_omp.h"
#include "timer.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   start up LAMMPS
   allocate fundamental classes (memory, error, universe, input)
   parse input switches
   initialize communicators, screen & logfile output
   input is allocated at end after MPI info is setup
------------------------------------------------------------------------- */

LAMMPS::LAMMPS(int narg, char **arg, MPI_Comm communicator)
{
  memory = new Memory(this);
  error = new Error(this);
  universe = new Universe(this,communicator);
  output = NULL;

  screen = NULL;
  logfile = NULL;

  // parse input switches

  int inflag = 0;
  int screenflag = 0;
  int logflag = 0;
  int partscreenflag = 0;
  int partlogflag = 0;
  int cudaflag = -1;
  int kokkosflag = -1;
  int restartflag = 0;
  int citeflag = 1;
  int helpflag = 0;

  suffix = NULL;
  suffix_enable = 0;
  char *rfile = NULL;
  char *dfile = NULL;
  int wdfirst,wdlast;
  int kkfirst,kklast;

  int iarg = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"-partition") == 0 ||
        strcmp(arg[iarg],"-p") == 0) {
      universe->existflag = 1;
      if (iarg+2 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      iarg++;
      while (iarg < narg && arg[iarg][0] != '-') {
        universe->add_world(arg[iarg]);
        iarg++;
      }
    } else if (strcmp(arg[iarg],"-in") == 0 ||
               strcmp(arg[iarg],"-i") == 0) {
      if (iarg+2 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      inflag = iarg + 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"-screen") == 0 ||
               strcmp(arg[iarg],"-sc") == 0) {
      if (iarg+2 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      screenflag = iarg + 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"-log") == 0 ||
               strcmp(arg[iarg],"-l") == 0) {
      if (iarg+2 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      logflag = iarg + 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"-var") == 0 ||
               strcmp(arg[iarg],"-v") == 0) {
      if (iarg+3 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      iarg += 3;
      while (iarg < narg && arg[iarg][0] != '-') iarg++;
    } else if (strcmp(arg[iarg],"-echo") == 0 ||
               strcmp(arg[iarg],"-e") == 0) {
      if (iarg+2 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      iarg += 2;
    } else if (strcmp(arg[iarg],"-pscreen") == 0 ||
               strcmp(arg[iarg],"-ps") == 0) {
      if (iarg+2 > narg)
       error->universe_all(FLERR,"Invalid command-line argument");
      partscreenflag = iarg + 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"-plog") == 0 ||
               strcmp(arg[iarg],"-pl") == 0) {
      if (iarg+2 > narg)
       error->universe_all(FLERR,"Invalid command-line argument");
      partlogflag = iarg + 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"-cuda") == 0 ||
               strcmp(arg[iarg],"-c") == 0) {
      if (iarg+2 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      if (strcmp(arg[iarg+1],"on") == 0) cudaflag = 1;
      else if (strcmp(arg[iarg+1],"off") == 0) cudaflag = 0;
      else error->universe_all(FLERR,"Invalid command-line argument");
      iarg += 2;
    } else if (strcmp(arg[iarg],"-kokkos") == 0 ||
               strcmp(arg[iarg],"-k") == 0) {
      if (iarg+2 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      if (strcmp(arg[iarg+1],"on") == 0) kokkosflag = 1;
      else if (strcmp(arg[iarg+1],"off") == 0) kokkosflag = 0;
      else error->universe_all(FLERR,"Invalid command-line argument");
      iarg += 2;
      // delimit any extra args for the Kokkos instantiation
      kkfirst = iarg;
      while (iarg < narg && arg[iarg][0] != '-') iarg++;
      kklast = iarg;
    } else if (strcmp(arg[iarg],"-suffix") == 0 ||
               strcmp(arg[iarg],"-sf") == 0) {
      if (iarg+2 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      delete [] suffix;
      int n = strlen(arg[iarg+1]) + 1;
      suffix = new char[n];
      strcpy(suffix,arg[iarg+1]);
      suffix_enable = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"-reorder") == 0 ||
               strcmp(arg[iarg],"-ro") == 0) {
      if (iarg+3 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      if (universe->existflag)
        error->universe_all(FLERR,"Cannot use -reorder after -partition");
      universe->reorder(arg[iarg+1],arg[iarg+2]);
      iarg += 3;
    } else if (strcmp(arg[iarg],"-restart") == 0 ||
               strcmp(arg[iarg],"-r") == 0) {
      if (iarg+3 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      restartflag = 1;
      rfile = arg[iarg+1];
      dfile = arg[iarg+2];
      iarg += 3;
      // delimit any extra args for the write_data command
      wdfirst = iarg;
      while (iarg < narg && arg[iarg][0] != '-') iarg++;
      wdlast = iarg;
    } else if (strcmp(arg[iarg],"-nocite") == 0 ||
               strcmp(arg[iarg],"-nc") == 0) {
      citeflag = 0;
      iarg++;
    } else if (strcmp(arg[iarg],"-help") == 0 ||
               strcmp(arg[iarg],"-h") == 0) {
      if (iarg+1 > narg)
        error->universe_all(FLERR,"Invalid command-line argument");
      helpflag = 1;
      iarg += 1;
    } else error->universe_all(FLERR,"Invalid command-line argument");
  }

  // if no partition command-line switch, universe is one world with all procs

  if (universe->existflag == 0) universe->add_world(NULL);

  // sum of procs in all worlds must equal total # of procs

  if (!universe->consistent())
    error->universe_all(FLERR,"Processor partitions are inconsistent");

  // universe cannot use stdin for input file

  if (universe->existflag && inflag == 0)
    error->universe_all(FLERR,"Must use -in switch with multiple partitions");

  // if no partition command-line switch, cannot use -pscreen option

  if (universe->existflag == 0 && partscreenflag)
    error->universe_all(FLERR,"Can only use -pscreen with multiple partitions");

  // if no partition command-line switch, cannot use -plog option

  if (universe->existflag == 0 && partlogflag)
    error->universe_all(FLERR,"Can only use -plog with multiple partitions");

  // set universe screen and logfile

  if (universe->me == 0) {
    if (screenflag == 0)
      universe->uscreen = stdout;
    else if (strcmp(arg[screenflag],"none") == 0)
      universe->uscreen = NULL;
    else {
      universe->uscreen = fopen(arg[screenflag],"w");
      if (universe->uscreen == NULL)
        error->universe_one(FLERR,"Cannot open universe screen file");
    }
    if (logflag == 0) {
      if (helpflag == 0) {
        universe->ulogfile = fopen("log.lammps","w");
        if (universe->ulogfile == NULL)
          error->universe_warn(FLERR,"Cannot open log.lammps for writing");
      }
    } else if (strcmp(arg[logflag],"none") == 0)
      universe->ulogfile = NULL;
    else {
      universe->ulogfile = fopen(arg[logflag],"w");
      if (universe->ulogfile == NULL)
        error->universe_one(FLERR,"Cannot open universe log file");
    }
  }

  if (universe->me > 0) {
    if (screenflag == 0) universe->uscreen = stdout;
    else universe->uscreen = NULL;
    universe->ulogfile = NULL;
  }

  // make universe and single world the same, since no partition switch
  // world inherits settings from universe
  // set world screen, logfile, communicator, infile
  // open input script if from file

  if (universe->existflag == 0) {
    screen = universe->uscreen;
    logfile = universe->ulogfile;
    world = universe->uworld;
    infile = NULL;

    if (universe->me == 0) {
      if (inflag == 0) infile = stdin;
      else infile = fopen(arg[inflag],"r");
      if (infile == NULL) {
        char str[128];
        sprintf(str,"Cannot open input script %s",arg[inflag]);
        error->one(FLERR,str);
      }
    }

    if (universe->me == 0) {
      if (screen) fprintf(screen,"LAMMPS (%s)\n",universe->version);
      if (logfile) fprintf(logfile,"LAMMPS (%s)\n",universe->version);
    }

  // universe is one or more worlds, as setup by partition switch
  // split universe communicator into separate world communicators
  // set world screen, logfile, communicator, infile
  // open input script

  } else {
    int me;
    MPI_Comm_split(universe->uworld,universe->iworld,0,&world);
    MPI_Comm_rank(world,&me);

    if (me == 0)
      if (partscreenflag == 0)
       if (screenflag == 0) {
         char str[32];
         sprintf(str,"screen.%d",universe->iworld);
         screen = fopen(str,"w");
         if (screen == NULL) error->one(FLERR,"Cannot open screen file");
       } else if (strcmp(arg[screenflag],"none") == 0)
         screen = NULL;
       else {
         char str[128];
         sprintf(str,"%s.%d",arg[screenflag],universe->iworld);
         screen = fopen(str,"w");
         if (screen == NULL) error->one(FLERR,"Cannot open screen file");
       }
      else if (strcmp(arg[partscreenflag],"none") == 0)
        screen = NULL;
      else {
        char str[128];
        sprintf(str,"%s.%d",arg[partscreenflag],universe->iworld);
        screen = fopen(str,"w");
        if (screen == NULL) error->one(FLERR,"Cannot open screen file");
      } else screen = NULL;

    if (me == 0)
      if (partlogflag == 0)
       if (logflag == 0) {
         char str[32];
         sprintf(str,"log.lammps.%d",universe->iworld);
         logfile = fopen(str,"w");
         if (logfile == NULL) error->one(FLERR,"Cannot open logfile");
       } else if (strcmp(arg[logflag],"none") == 0)
         logfile = NULL;
       else {
         char str[128];
         sprintf(str,"%s.%d",arg[logflag],universe->iworld);
         logfile = fopen(str,"w");
         if (logfile == NULL) error->one(FLERR,"Cannot open logfile");
       }
      else if (strcmp(arg[partlogflag],"none") == 0)
        logfile = NULL;
      else {
        char str[128];
        sprintf(str,"%s.%d",arg[partlogflag],universe->iworld);
        logfile = fopen(str,"w");
        if (logfile == NULL) error->one(FLERR,"Cannot open logfile");
      } else logfile = NULL;

    if (me == 0) {
      infile = fopen(arg[inflag],"r");
      if (infile == NULL) {
        char str[128];
        sprintf(str,"Cannot open input script %s",arg[inflag]);
        error->one(FLERR,str);
      }
    } else infile = NULL;

    // screen and logfile messages for universe and world

    if (universe->me == 0) {
      if (universe->uscreen) {
        fprintf(universe->uscreen,"LAMMPS (%s)\n",universe->version);
        fprintf(universe->uscreen,"Running on %d partitions of processors\n",
                universe->nworlds);
      }
      if (universe->ulogfile) {
        fprintf(universe->ulogfile,"LAMMPS (%s)\n",universe->version);
        fprintf(universe->ulogfile,"Running on %d partitions of processors\n",
                universe->nworlds);
      }
    }

    if (me == 0) {
      if (screen) {
        fprintf(screen,"LAMMPS (%s)\n",universe->version);
        fprintf(screen,"Processor partition = %d\n",universe->iworld);
      }
      if (logfile) {
        fprintf(logfile,"LAMMPS (%s)\n",universe->version);
        fprintf(logfile,"Processor partition = %d\n",universe->iworld);
      }
    }
  }

  // check consistency of datatype settings in lmptype.h

  if (sizeof(smallint) != sizeof(int))
    error->all(FLERR,"Smallint setting in lmptype.h is invalid");
  if (sizeof(imageint) < sizeof(smallint))
    error->all(FLERR,"Imageint setting in lmptype.h is invalid");
  if (sizeof(tagint) < sizeof(smallint))
    error->all(FLERR,"Tagint setting in lmptype.h is invalid");
  if (sizeof(bigint) < sizeof(imageint) || sizeof(bigint) < sizeof(tagint))
    error->all(FLERR,"Bigint setting in lmptype.h is invalid");

  int mpisize;
  MPI_Type_size(MPI_LMP_TAGINT,&mpisize);
  if (mpisize != sizeof(tagint))
      error->all(FLERR,"MPI_LMP_TAGINT and tagint in "
                 "lmptype.h are not compatible");
  MPI_Type_size(MPI_LMP_BIGINT,&mpisize);
  if (mpisize != sizeof(bigint))
      error->all(FLERR,"MPI_LMP_BIGINT and bigint in "
                 "lmptype.h are not compatible");

#ifdef LAMMPS_SMALLBIG
  if (sizeof(smallint) != 4 || sizeof(imageint) != 4 || 
      sizeof(tagint) != 4 || sizeof(bigint) != 8)
    error->all(FLERR,"Small to big integers are not sized correctly");
#endif
#ifdef LAMMPS_BIGBIG
  if (sizeof(smallint) != 4 || sizeof(imageint) != 8 || 
      sizeof(tagint) != 8 || sizeof(bigint) != 8)
    error->all(FLERR,"Small to big integers are not sized correctly");
#endif
#ifdef LAMMPS_SMALLSMALL
  if (sizeof(smallint) != 4 || sizeof(imageint) != 4 || 
      sizeof(tagint) != 4 || sizeof(bigint) != 4)
    error->all(FLERR,"Small to big integers are not sized correctly");
#endif

  // error check on accelerator packages

  if (cudaflag == 1 && kokkosflag == 1) 
    error->all(FLERR,"Cannot use -cuda on and -kokkos on");

  // create Cuda class if USER-CUDA installed, unless explicitly switched off
  // instantiation creates dummy Cuda class if USER-CUDA is not installed

  if (cudaflag == 0) {
    cuda = NULL;
  } else if (cudaflag == 1) {
    cuda = new Cuda(this);
    if (!cuda->cuda_exists)
      error->all(FLERR,"Cannot use -cuda on without USER-CUDA installed");
  } else {
    cuda = new Cuda(this);
    if (!cuda->cuda_exists) {
      delete cuda;
      cuda = NULL;
    }
  }

  int me;
  MPI_Comm_rank(world,&me);
  if (cuda && me == 0) error->message(FLERR,"USER-CUDA mode is enabled");

  // create Kokkos class if KOKKOS installed, unless explicitly switched off
  // instantiation creates dummy Kokkos class if KOKKOS is not installed
  // add args between kkfirst and kklast to Kokkos instantiation

  if (kokkosflag == 0) {
    kokkos = NULL;
  } else if (kokkosflag == 1) {
    kokkos = new KokkosLMP(this,kklast-kkfirst,&arg[kkfirst]);
    if (!kokkos->kokkos_exists)
      error->all(FLERR,"Cannot use -kokkos on without KOKKOS installed");
  } else {
    kokkos = new KokkosLMP(this,kklast-kkfirst,&arg[kkfirst]);
    if (!kokkos->kokkos_exists) {
      delete kokkos;
      kokkos = NULL;
    }
  }

  MPI_Comm_rank(world,&me);
  if (kokkos && me == 0) error->message(FLERR,"KOKKOS mode is enabled");

  // allocate CiteMe class if enabled

  if (citeflag) citeme = new CiteMe(this);
  else citeme = NULL;

  // allocate input class now that MPI is fully setup

  input = new Input(this,narg,arg);

  // allocate top-level classes

  create();
  post_create();

  // if helpflag set, print help and quit

  if (helpflag) {
    if (universe->me == 0 && screen) help();
    error->done();
  }

  // if restartflag set, invoke 2 commands and quit
  // add args between wdfirst and wdlast to write_data command
  // also add "noinit" to prevent write_data from doing system init

  if (restartflag) {
    char cmd[128];
    sprintf(cmd,"read_restart %s\n",rfile);
    input->one(cmd);
    sprintf(cmd,"write_data %s",dfile);
    for (iarg = wdfirst; iarg < wdlast; iarg++)
      sprintf(&cmd[strlen(cmd)]," %s",arg[iarg]);
    strcat(cmd," noinit\n");
    input->one(cmd);
    error->done();
  }
}

/* ----------------------------------------------------------------------
   shutdown LAMMPS
   delete top-level classes
   close screen and log files in world and universe
   output files were already closed in destroy()
   delete fundamental classes
------------------------------------------------------------------------- */

LAMMPS::~LAMMPS()
{
  destroy();

  delete citeme;

  if (universe->nworlds == 1) {
    if (logfile) fclose(logfile);
  } else {
    if (screen && screen != stdout) fclose(screen);
    if (logfile) fclose(logfile);
    if (universe->ulogfile) fclose(universe->ulogfile);
  }

  if (world != universe->uworld) MPI_Comm_free(&world);

  delete cuda;
  delete kokkos;
  delete [] suffix;

  delete input;
  delete universe;
  delete error;
  delete memory;
}

/* ----------------------------------------------------------------------
   allocate single instance of top-level classes
   fundamental classes are allocated in constructor
   some classes have package variants
------------------------------------------------------------------------- */

void LAMMPS::create()
{
  // Comm class must be created before Atom class
  // so that nthreads is defined when create_avec invokes grow()

  if (cuda) comm = new CommCuda(this);
  else if (kokkos) comm = new CommKokkos(this);
  else comm = new Comm(this);

  if (cuda) neighbor = new NeighborCuda(this);
  else if (kokkos) neighbor = new NeighborKokkos(this);
  else neighbor = new Neighbor(this);

  if (cuda) domain = new DomainCuda(this);
  else if (kokkos) domain = new DomainKokkos(this);
#ifdef LMP_USER_OMP
  else domain = new DomainOMP(this);
#else
  else domain = new Domain(this);
#endif

  if (kokkos) atom = new AtomKokkos(this);
  else atom = new Atom(this);
  atom->create_avec("atomic",0,NULL,suffix);

  group = new Group(this);
  force = new Force(this);    // must be after group, to create temperature

  if (cuda) modify = new ModifyCuda(this);
  else if (kokkos) modify = new ModifyKokkos(this);
  else modify = new Modify(this);

  output = new Output(this);  // must be after group, so "all" exists
                              // must be after modify so can create Computes
  update = new Update(this);  // must be after output, force, neighbor
  timer = new Timer(this);
}

/* ----------------------------------------------------------------------
   invoke package-specific setup commands
   called from LAMMPS constructor and after clear() command
   only invoke if suffix is set and enabled
------------------------------------------------------------------------- */

void LAMMPS::post_create()
{
  if (suffix && suffix_enable) {
    if (strcmp(suffix,"gpu") == 0) input->one("package gpu force/neigh 0 0 1");
    if (strcmp(suffix,"omp") == 0) input->one("package omp *");
  }
}

/* ----------------------------------------------------------------------
   initialize top-level classes
   do not initialize Timer class, other classes like Run() do that explicitly
------------------------------------------------------------------------- */

void LAMMPS::init()
{
  if (cuda) cuda->accelerator(0,NULL);
  if (kokkos) kokkos->accelerator(0,NULL);

  update->init();
  force->init();         // pair must come after update due to minimizer
  domain->init();
  atom->init();          // atom must come after force and domain
                         //   atom deletes extra array
                         //   used by fix shear_history::unpack_restart()
                         //   when force->pair->gran_history creates fix ??
                         //   atom_vec init uses deform_vremap
  modify->init();        // modify must come after update, force, atom, domain
  neighbor->init();      // neighbor must come after force, modify
  comm->init();          // comm must come after force, modify, neighbor, atom
  output->init();        // output must come after domain, force, modify
}

/* ----------------------------------------------------------------------
   delete single instance of top-level classes
   fundamental classes are deleted in destructor
------------------------------------------------------------------------- */

void LAMMPS::destroy()
{
  delete update;
  delete neighbor;
  delete comm;
  delete force;
  delete group;
  delete output;

  delete modify;          // modify must come after output, force, update
                          //   since they delete fixes
  delete domain;          // domain must come after modify
                          //   since fix destructors access domain
  delete atom;            // atom must come after modify, neighbor
                          //   since fixes delete callbacks in atom
  delete timer;

  modify = NULL;          // necessary since input->variable->varreader
                          // will be destructed later
}

/* ----------------------------------------------------------------------
   help message for command line options and styles present in executable
------------------------------------------------------------------------- */

void LAMMPS::help()
{
  fprintf(screen,
          "\nCommand line options:\n\n"
          "-cuda on/off                : turn CUDA mode on or off (-c)\n"
          "-echo none/screen/log/both  : echoing of input script (-e)\n"
          "-in filename                : read input from file, not stdin (-i)\n"
          "-help                       : print this help message (-h)\n"
          "-kokkos on/off ...          : turn KOKKOS mode on or off (-k)\n"
          "-log none/filename          : where to send log output (-l)\n"
          "-nocite                     : disable writing log.cite file (-nc)\n"
          "-partition size1 size2 ...  : assign partition sizes (-p)\n"
          "-plog basename              : basename for partition logs (-pl)\n"
          "-pscreen basename           : basename for partition screens (-ps)\n"
          "-reorder topology-specs     : processor reordering (-r)\n"
          "-screen none/filename       : where to send screen output (-sc)\n"
          "-suffix cuda/gpu/opt/omp    : style suffix to apply (-sf)\n"
          "-var varname value          : set index style variable (-v)\n\n");
  
  fprintf(screen,"Style options compiled with this executable\n\n");

  int pos = 80;
  fprintf(screen,"* Atom styles:\n");
#define ATOM_CLASS
#define AtomStyle(key,Class) print_style(#key,pos);
#include "style_atom.h"
#undef ATOM_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Integrate styles:\n");
#define INTEGRATE_CLASS
#define IntegrateStyle(key,Class) print_style(#key,pos);
#include "style_integrate.h"
#undef INTEGRATE_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Minimize styles:\n");
#define MINIMIZE_CLASS
#define MinimizeStyle(key,Class) print_style(#key,pos);
#include "style_minimize.h"
#undef MINIMIZE_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Pair styles:\n");
#define PAIR_CLASS
#define PairStyle(key,Class) print_style(#key,pos);
#include "style_pair.h"
#undef PAIR_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Bond styles:\n");
#define BOND_CLASS
#define BondStyle(key,Class) print_style(#key,pos);
#include "style_bond.h"
#undef BOND_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Angle styles:\n");
#define ANGLE_CLASS
#define AngleStyle(key,Class) print_style(#key,pos);
#include "style_angle.h"
#undef ANGLE_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Dihedral styles:\n");
#define DIHEDRAL_CLASS
#define DihedralStyle(key,Class) print_style(#key,pos);
#include "style_dihedral.h"
#undef DIHEDRAL_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Improper styles:\n");
#define IMPROPER_CLASS
#define ImproperStyle(key,Class) print_style(#key,pos);
#include "style_improper.h"
#undef IMPROPER_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* KSpace styles:\n");
#define KSPACE_CLASS
#define KSpaceStyle(key,Class) print_style(#key,pos);
#include "style_kspace.h"
#undef KSPACE_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Fix styles\n");
#define FIX_CLASS
#define FixStyle(key,Class) print_style(#key,pos);
#include "style_fix.h"
#undef FIX_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Compute styles:\n");
#define COMPUTE_CLASS
#define ComputeStyle(key,Class) print_style(#key,pos);
#include "style_compute.h"
#undef COMPUTE_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Region styles:\n");
#define REGION_CLASS
#define RegionStyle(key,Class) print_style(#key,pos);
#include "style_region.h"
#undef REGION_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Dump styles:\n");
#define DUMP_CLASS
#define DumpStyle(key,Class) print_style(#key,pos);
#include "style_dump.h"
#undef DUMP_CLASS
  fprintf(screen,"\n\n");

  pos = 80;
  fprintf(screen,"* Command styles\n");
#define COMMAND_CLASS
#define CommandStyle(key,Class) print_style(#key,pos);
#include "style_command.h"
#undef COMMAND_CLASS
  fprintf(screen,"\n");
}

/* ----------------------------------------------------------------------
   print style names in columns
   skip any style that starts with upper-case letter, since internal
------------------------------------------------------------------------- */

void LAMMPS::print_style(const char *str, int &pos)
{
  if (isupper(str[0])) return;

  int len = strlen(str);
  if (pos+len > 80) { 
    fprintf(screen,"\n");
    pos = 0;
  }

  if (len < 16) {
    fprintf(screen,"%-16s",str);
    pos += 16;
  } else if (len < 32) {
    fprintf(screen,"%-32s",str);
    pos += 32;
  } else if (len < 48) {
    fprintf(screen,"%-48s",str);
    pos += 48;
  } else if (len < 64) {
    fprintf(screen,"%-64s",str);
    pos += 64;
  } else {
    fprintf(screen,"%-80s",str);
    pos += 80;
  }
}

#if 0
#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM

#include "analytics_kmeans_cuda_kernel.cu"

//#define BLOCK_DELTA_REDUCE
//#define BLOCK_CENTER_REDUCE

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;                                      /* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;                                        /* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;       /* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;          /* number of blocks */

/* _d denotes it resides on the device */
int    *membership_new;                                                                                         /* newly assignment membership */
float  *feature_d;                                                                                                      /* inverted data array */
float  *feature_flipped_d;                                                                                      /* original (not inverted) data array */
int    *membership_d;                                                                                           /* membership on the device */
float  *block_new_centers;                                                                                      /* sum of points in a cluster (per block) */
float  *clusters_d;                                                                                                     /* cluster centers on the device */
float  *block_clusters_d;                                                                                       /* per block calculation of cluster centers */
int    *block_deltas_d;                                                                                         /* per block calculation of deltas */

/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
extern "C"
void allocateMemory(int npoints, int nfeatures, int nclusters, float **features)
{
        num_blocks = npoints / num_threads;
        if (npoints % num_threads > 0)          /* defeat truncation */
                num_blocks++;

        num_blocks_perdim = sqrt((double) num_blocks);
        while (num_blocks_perdim * num_blocks_perdim < num_blocks)      // defeat truncation (should run once)
                num_blocks_perdim++;

        num_blocks = num_blocks_perdim*num_blocks_perdim;

        /* allocate memory for memory_new[] and initialize to -1 (host) */
        membership_new = (int*) malloc(npoints * sizeof(int));
        for(int i=0;i<npoints;i++) {
                membership_new[i] = -1;
        }

        /* allocate memory for block_new_centers[] (host) */
        block_new_centers = (float *) malloc(nclusters*nfeatures*sizeof(float));

        /* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
        cudaMalloc((void**) &feature_flipped_d, npoints*nfeatures*sizeof(float));
        cudaMemcpy(feature_flipped_d, features[0], npoints*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void**) &feature_d, npoints*nfeatures*sizeof(float));

        /* invert the data array (kernel execution) */
        invert_mapping<<<num_blocks,num_threads>>>(feature_flipped_d,feature_d,npoints,nfeatures);

        /* allocate memory for membership_d[] and clusters_d[][] (device) */
        cudaMalloc((void**) &membership_d, npoints*sizeof(int));
        cudaMalloc((void**) &clusters_d, nclusters*nfeatures*sizeof(float));


#ifdef BLOCK_DELTA_REDUCE
        // allocate array to hold the per block deltas on the gpu side

        cudaMalloc((void**) &block_deltas_d, num_blocks_perdim * num_blocks_perdim * sizeof(int));
        //cudaMemcpy(block_delta_d, &delta_h, sizeof(int), cudaMemcpyHostToDevice);
#endif

#ifdef BLOCK_CENTER_REDUCE
        // allocate memory and copy to card cluster  array in which to accumulate center points for the next iteration
        cudaMalloc((void**) &block_clusters_d,
        num_blocks_perdim * num_blocks_perdim *
        nclusters * nfeatures * sizeof(float));
        //cudaMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
#endif

}
/* -------------- allocateMemory() end ------------------- */

/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
extern "C"
void deallocateMemory()
{
        free(membership_new);
        free(block_new_centers);
        cudaFree(feature_d);
        cudaFree(feature_flipped_d);
        cudaFree(membership_d);

        cudaFree(clusters_d);
#ifdef BLOCK_CENTER_REDUCE
    cudaFree(block_clusters_d);
#endif
#ifdef BLOCK_DELTA_REDUCE
    cudaFree(block_deltas_d);
#endif
}
/* -------------- deallocateMemory() end ------------------- */

/* ------------------- kmeansCuda() ------------------------ */
int     // delta -- had problems when return value was of float type
kmeansCuda(float  **feature,                            /* in: [npoints][nfeatures] */
           int      nfeatures,                          /* number of attributes for each point */
           int      npoints,                            /* number of data points */
           int      nclusters,                          /* number of clusters */
           int     *membership,                         /* which cluster the point belongs to */
                   float  **clusters,                           /* coordinates of cluster centers */
                   int     *new_centers_len,            /* number of elements in each cluster */
           float  **new_centers                         /* sum of elements in each cluster */
                   )
{
        int delta = 0;                  /* if point has moved */
        int i,j;                                /* counters */


        cudaSetDevice(1);

        /* copy membership (host to device) */
        cudaMemcpy(membership_d, membership_new, npoints*sizeof(int), cudaMemcpyHostToDevice);

        /* copy clusters (host to device) */
        cudaMemcpy(clusters_d, clusters[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);

        /* set up texture */
    cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
    t_features.filterMode = cudaFilterModePoint;
    t_features.normalized = false;
    t_features.channelDesc = chDesc0;

        if(cudaBindTexture(NULL, &t_features, feature_d, &chDesc0, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
        printf("Couldn't bind features array to texture!\n");

        cudaChannelFormatDesc chDesc1 = cudaCreateChannelDesc<float>();
    t_features_flipped.filterMode = cudaFilterModePoint;
    t_features_flipped.normalized = false;
    t_features_flipped.channelDesc = chDesc1;

        if(cudaBindTexture(NULL, &t_features_flipped, feature_flipped_d, &chDesc1, npoints*nfeatures*sizeof(float)) != CUDA_SUCCESS)
        printf("Couldn't bind features_flipped array to texture!\n");

        cudaChannelFormatDesc chDesc2 = cudaCreateChannelDesc<float>();
    t_clusters.filterMode = cudaFilterModePoint;
    t_clusters.normalized = false;
    t_clusters.channelDesc = chDesc2;

        if(cudaBindTexture(NULL, &t_clusters, clusters_d, &chDesc2, nclusters*nfeatures*sizeof(float)) != CUDA_SUCCESS)
        printf("Couldn't bind clusters array to texture!\n");

        /* copy clusters to constant memory */
        cudaMemcpyToSymbol("c_clusters",clusters[0],nclusters*nfeatures*sizeof(float),0,cudaMemcpyHostToDevice);


    /* setup execution parameters.
           changed to 2d (source code on NVIDIA CUDA Programming Guide) */
    dim3  grid( num_blocks_perdim, num_blocks_perdim );
    dim3  threads( num_threads_perdim*num_threads_perdim );

        /* execute the kernel */
    kmeansPoint<<< grid, threads >>>( feature_d,
                                      nfeatures,
                                      npoints,
                                      nclusters,
                                      membership_d,
                                      clusters_d,
                                                                          block_clusters_d,
                                                                          block_deltas_d);

        cudaThreadSynchronize();

        /* copy back membership (device to host) */
        cudaMemcpy(membership_new, membership_d, npoints*sizeof(int), cudaMemcpyDeviceToHost);

#ifdef BLOCK_CENTER_REDUCE
    /*** Copy back arrays of per block sums ***/
    float * block_clusters_h = (float *) malloc(
        num_blocks_perdim * num_blocks_perdim *
        nclusters * nfeatures * sizeof(float));

        cudaMemcpy(block_clusters_h, block_clusters_d,
        num_blocks_perdim * num_blocks_perdim *
        nclusters * nfeatures * sizeof(float),
        cudaMemcpyDeviceToHost);
#endif
#ifdef BLOCK_DELTA_REDUCE
    int * block_deltas_h = (int *) malloc(
        num_blocks_perdim * num_blocks_perdim * sizeof(int));

        cudaMemcpy(block_deltas_h, block_deltas_d,
        num_blocks_perdim * num_blocks_perdim * sizeof(int),
        cudaMemcpyDeviceToHost);
#endif

        /* for each point, sum data points in each cluster
           and see if membership has changed:
             if so, increase delta and change old membership, and update new_centers;
             otherwise, update new_centers */
        delta = 0;
        for (i = 0; i < npoints; i++)
        {
                int cluster_id = membership_new[i];
                new_centers_len[cluster_id]++;
                if (membership_new[i] != membership[i])
                {
#ifdef CPU_DELTA_REDUCE
                        delta++;
#endif
                        membership[i] = membership_new[i];
                }
#ifdef CPU_CENTER_REDUCE
                for (j = 0; j < nfeatures; j++)
                {
                        new_centers[cluster_id][j] += feature[i][j];
                }
#endif
        }


#ifdef BLOCK_DELTA_REDUCE       
    /*** calculate global sums from per block sums for delta and the new centers ***/

        //debug
        //printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim * num_blocks_perdim);
    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
                //printf("block %d delta is %d \n",i,block_deltas_h[i]);
        delta += block_deltas_h[i];
    }

#endif
#ifdef BLOCK_CENTER_REDUCE      

        for(int j = 0; j < nclusters;j++) {
                for(int k = 0; k < nfeatures;k++) {
                        block_new_centers[j*nfeatures + k] = 0.f;
                }
        }

    for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
                for(int j = 0; j < nclusters;j++) {
                        for(int k = 0; k < nfeatures;k++) {
                                block_new_centers[j*nfeatures + k] += block_clusters_h[i * nclusters*nfeatures + j * nfeatures + k];
                        }
                }
    }


#ifdef CPU_CENTER_REDUCE
        //debug
        /*for(int j = 0; j < nclusters;j++) {
                for(int k = 0; k < nfeatures;k++) {
                        if(new_centers[j][k] >  1.001 * block_new_centers[j*nfeatures + k] || new_centers[j][k] <       0.999 * block_new_centers[j*nfeatures + k]) {
                                printf("\t \t for %d:%d, normal value is %e and gpu reduced value id %e \n",j,k,new_centers[j][k],block_new_centers[j*nfeatures + k]);
                        }
                }
        }*/
#endif

#ifdef BLOCK_CENTER_REDUCE
        for(int j = 0; j < nclusters;j++) {
                for(int k = 0; k < nfeatures;k++)
                        new_centers[j][k]= block_new_centers[j*nfeatures + k];
        }
#endif

#endif

        return delta;

}
/* ------------------- kmeansCuda() end ------------------------ */

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline
float euclid_dist_2(float *pt1,
                    float *pt2,
                    int    numdims)
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (pt1[i]-pt2[i]) * (pt1[i]-pt2[i]);

    return(ans);
}

/*----< find_nearest_point() >-----------------------------------------------*/
__inline
int find_nearest_point(float  *pt,          /* [nfeatures] */
                       int     nfeatures,
                       float  **pts,         /* [npts][nfeatures] */
                       int     npts)
{
    int index, i;
    float max_dist=FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i=0; i<npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
        if (dist < max_dist) {
            max_dist = dist;
            index    = i;
        }
    }
    return(index);
}

/*----< rms_err(): calculates RMSE of clustering >-------------------------------------*/
float rms_err   (float **feature,         /* [npoints][nfeatures] */
                 int     nfeatures,
                 int     npoints,
                 float **cluster_centres, /* [nclusters][nfeatures] */
                 int     nclusters)
{
    int    i;
        int        nearest_cluster_index;       /* cluster center id with min distance to pt */
    float  sum_euclid = 0.0;            /* sum of Euclidean distance squares */
    float  ret;                                         /* return value */

    /* calculate and sum the sqaure of euclidean distance*/
    #pragma omp parallel for \
                shared(feature,cluster_centres) \
                firstprivate(npoints,nfeatures,nclusters) \
                private(i, nearest_cluster_index) \
                schedule (static)       
    for (i=0; i<npoints; i++) {
        nearest_cluster_index = find_nearest_point(feature[i],
                                                                                                        nfeatures,
                                                                                                        cluster_centres,
                                                                                                        nclusters);

                sum_euclid += euclid_dist_2(feature[i],
                                                                        cluster_centres[nearest_cluster_index],
                                                                        nfeatures);

    }
        /* divide by n, then take sqrt */
        ret = sqrt(sum_euclid / npoints);

    return(ret);
}

/*----< kmeans_clustering() >---------------------------------------------*/
float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
                          int     nfeatures,
                          int     npoints,
                          int     nclusters,
                          float   threshold,
                          int    *membership) /* out: [npoints] */
{
    int      i, j, n = 0;                               /* counters */
        int              loop=0, temp;
    int     *new_centers_len;   /* [nclusters]: no. of points in each cluster */
    float    delta;                             /* if the point moved */
    float  **clusters;                  /* out: [nclusters][nfeatures] */
    float  **new_centers;               /* [nclusters][nfeatures] */

        int     *initial;                       /* used to hold the index of points not yet selected
                                                                   prevents the "birthday problem" of dual selection (?)
                                                                   considered holding initial cluster indices, but changed due to
                                                                   possible, though unlikely, infinite loops */
        int      initial_points;
        int              c = 0;

        /* nclusters should never be > npoints
           that would guarantee a cluster without points */
        if (nclusters > npoints)
                nclusters = npoints;

    /* allocate space for and initialize returning variable clusters[] */
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (i=1; i<nclusters; i++)
        clusters[i] = clusters[i-1] + nfeatures;

        /* initialize the random clusters */
        initial = (int *) malloc (npoints * sizeof(int));
        for (i = 0; i < npoints; i++)
        {
                initial[i] = i;
        }
        initial_points = npoints;

    /* randomly pick cluster centers */
    for (i=0; i<nclusters && initial_points >= 0; i++) {
                //n = (int)rand() % initial_points;             

        for (j=0; j<nfeatures; j++)
            clusters[i][j] = feature[initial[n]][j];    // remapped

                /* swap the selected index to the end (not really necessary,
                   could just move the end up) */
                temp = initial[n];
                initial[n] = initial[initial_points-1];
                initial[initial_points-1] = temp;
                initial_points--;
                n++;
    }

        /* initialize the membership to -1 for all */
    for (i=0; i < npoints; i++)
          membership[i] = -1;

    /* allocate space for and initialize new_centers_len and new_centers */
    new_centers_len = (int*) calloc(nclusters, sizeof(int));

    new_centers    = (float**) malloc(nclusters *            sizeof(float*));
    new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
    for (i=1; i<nclusters; i++)
        new_centers[i] = new_centers[i-1] + nfeatures;

        /* iterate until convergence */
        do {
        delta = 0.0;
                // CUDA
                delta = (float) kmeansCuda(feature,                     /* in: [npoints][nfeatures] */
                                                                   nfeatures,           /* number of attributes for each point */
                                                                   npoints,                     /* number of data points */
                                                                   nclusters,           /* number of clusters */
                                                                   membership,          /* which cluster the point belongs to */
                                                                   clusters,            /* out: [nclusters][nfeatures] */
                                                                   new_centers_len,     /* out: number of points in each cluster */
                                                                   new_centers          /* sum of points in each cluster */
                                                                   );

                /* replace old cluster centers with new_centers */
                /* CPU side of reduction */
                for (i=0; i<nclusters; i++) {
                        for (j=0; j<nfeatures; j++) {
                                if (new_centers_len[i] > 0)
                                        clusters[i][j] = new_centers[i][j] / new_centers_len[i];        /* take average i.e. sum/n */
                                new_centers[i][j] = 0.0;        /* set back to 0 */
                        }
                        new_centers_len[i] = 0;                 /* set back to 0 */
                }
                c++;
    } while ((delta > threshold) && (loop++ < 500));    /* makes sure loop terminates */
        printf("iterated %d times\n", c);
    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}


float   min_rmse_ref = FLT_MAX;                 /* reference min_rmse value */

/*---< cluster() >-----------------------------------------------------------*/
int cluster(int      npoints,                           /* number of data points */
            int      nfeatures,                         /* number of attributes for each point */
            float  **features,                  /* array: [npoints][nfeatures] */
            int      min_nclusters,                     /* range of min to max number of clusters */
                        int              max_nclusters,
            float    threshold,                         /* loop terminating factor */
            int     *best_nclusters,            /* out: number between min and max with lowest RMSE */
            float ***cluster_centres,           /* out: [best_nclusters][nfeatures] */
                        float   *min_rmse,                              /* out: minimum RMSE */
                        int              isRMSE,                                /* calculate RMSE */
                        int              nloops                                 /* number of iteration for each number of clusters */
                        )
{
        int             nclusters;                                              /* number of clusters k */
        int             index =0;                                               /* number of iteration to reach the best RMSE */
        int             rmse;                                                   /* RMSE for each clustering */
    int    *membership;                                         /* which cluster a data point belongs to */
    float **tmp_cluster_centres;                        /* hold coordinates of cluster centers */
        int             i;

        /* allocate memory for membership */
    membership = (int*) malloc(npoints * sizeof(int));

        /* sweep k from min to max_nclusters to find the best number of clusters */
        for(nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
        {
                if (nclusters > npoints) break; /* cannot have more clusters than points */

                /* allocate device memory, invert data array (@ kmeans_cuda.cu) */
                allocateMemory(npoints, nfeatures, nclusters, features);

                /* iterate nloops times for each number of clusters */
                for(i = 0; i < nloops; i++)
                {
                        /* initialize initial cluster centers, CUDA calls (@ kmeans_cuda.cu) */
                        tmp_cluster_centres = kmeans_clustering(features,
                                                                                                        nfeatures,
                                                                                                        npoints,
                                                                                                        nclusters,
                                                                                                        threshold,
                                                                                                        membership);

                        if (*cluster_centres) {
                                free((*cluster_centres)[0]);
                                free(*cluster_centres);
                        }
                        *cluster_centres = tmp_cluster_centres;

                        /* find the number of clusters with the best RMSE */
                        if(isRMSE)
                        {
                                rmse = rms_err(features,
                                                           nfeatures,
                                                           npoints,
                                                           tmp_cluster_centres,
                                                           nclusters);

                                if(rmse < min_rmse_ref){
                                        min_rmse_ref = rmse;                    //update reference min RMSE
                                        *min_rmse = min_rmse_ref;               //update return min RMSE
                                        *best_nclusters = nclusters;    //update optimum number of clusters
                                        index = i;                                              //update number of iteration to reach best RMSE
                                }
                        }
                }

                deallocateMemory();                                                     /* free device memory (@ kmeans_cuda.cu) */
        }

    free(membership);

    return index;
}


/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
    char *help =
        "\nUsage: %s [switches] -i filename\n\n"
                "    -i filename      :file containing data to be clustered\n"
                "    -m max_nclusters :maximum number of clusters allowed    [default=5]\n"
        "    -n min_nclusters :minimum number of clusters allowed    [default=5]\n"
                "    -t threshold     :threshold value                       [default=0.001]\n"
                "    -l nloops        :iteration for each number of clusters [default=1]\n"
                "    -b               :input file is in binary format\n"
        "    -r               :calculate RMSE                        [default=off]\n"
                "    -o               :output cluster center coordinates     [default=off]\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int setup(int argc, char **argv) {
                int             opt;
 extern char   *optarg;
                char   *filename = 0;
                float  *buf;
                char    line[1024];
                int             isBinaryFile = 0;

                float   threshold = 0.001;              /* default value */
                int             max_nclusters=5;                /* default value */
                int             min_nclusters=5;                /* default value */
                int             best_nclusters = 0;
                int             nfeatures = 0;
                int             npoints = 0;
                float   len;

                float **features;
                float **cluster_centres=NULL;
                int             i, j, index;
                int             nloops = 1;                             /* default value */

                int             isRMSE = 0;
                float   rmse;

                int             isOutput = 0;
                //float cluster_timing, io_timing;              

                /* obtain command line arguments and change appropriate options */
                while ( (opt=getopt(argc,argv,"i:t:m:n:l:bro"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'm': max_nclusters = atoi(optarg);
                      break;
            case 'n': min_nclusters = atoi(optarg);
                      break;
                        case 'r': isRMSE = 1;
                      break;
                        case 'o': isOutput = 1;
                                          break;
                    case 'l': nloops = atoi(optarg);
                                          break;
            case '?': usage(argv[0]);
                      break;
            default: usage(argv[0]);
                      break;
        }
    }

    if (filename == 0) usage(argv[0]);

        // Initialize MPI state
        //MPI_CHECK(MPI_Init(&argc, &argv));

        // Get our MPI node number and node count
        //int commSize, commRank;
        //MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
        //MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

        //fprintf(stdout, "Node %d of %d\n", commRank, commSize);

        /* ============== I/O begin ==============*/
    /* get nfeatures and npoints */
    //io_timing = omp_get_wtime();
    if (isBinaryFile) {         //Binary file input
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
        }
        read(infile, &npoints,   sizeof(int));
        read(infile, &nfeatures, sizeof(int));

        /* allocate space for features[][] and read attributes of all objects */
        buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
        features    = (float**)malloc(npoints*          sizeof(float*));
        features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
        for (i=1; i<npoints; i++)
            features[i] = features[i-1] + nfeatures;

        read(infile, buf, npoints*nfeatures*sizeof(float));
        close(infile);
    }
    else {
        FILE *infile;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            exit(1);
                }
        while (fgets(line, 1024, infile) != NULL)
                        if (strtok(line, " \t\n") != 0)
                npoints++;
        rewind(infile);
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first attribute): nfeatures = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) nfeatures++;
                break;
            }
        }

        /* allocate space for features[] and read attributes of all objects */
        buf         = (float*) malloc(npoints*nfeatures*sizeof(float));
        features    = (float**)malloc(npoints*          sizeof(float*));
        features[0] = (float*) malloc(npoints*nfeatures*sizeof(float));
        for (i=1; i<npoints; i++)
            features[i] = features[i-1] + nfeatures;
        rewind(infile);
        i = 0;
        while (fgets(line, 1024, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<nfeatures; j++) {
                buf[i] = atof(strtok(NULL, " ,\t\n"));
                i++;
            }
        }
        fclose(infile);
    }
    //io_timing = omp_get_wtime() - io_timing;

        printf("\nI/O completed\n");
        printf("\nNumber of objects: %d\n", npoints);
        printf("Number of features: %d\n", nfeatures);
        /* ============== I/O end ==============*/

        // error check for clusters
        if (npoints < min_nclusters)
        {
                printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", min_nclusters, npoints);
                exit(0);
        }

        srand(7);                                                                                               /* seed for future random number generator */
        memcpy(features[0], buf, npoints*nfeatures*sizeof(float)); /* now features holds 2-dimensional array of features */
        free(buf);

        /* ======================= core of the clustering ===================*/
    //cluster_timing = omp_get_wtime();         /* Total clustering time */
        cluster_centres = NULL;
    index = cluster(npoints,                            /* number of data points */
                                        nfeatures,                              /* number of features for each point */
                                        features,                               /* array: [npoints][nfeatures] */
                                        min_nclusters,                  /* range of min to max number of clusters */
                                        max_nclusters,
                                        threshold,                              /* loop termination factor */
                                   &best_nclusters,                     /* return: number between min and max */
                                   &cluster_centres,            /* return: [best_nclusters][nfeatures] */
                                   &rmse,                                       /* Root Mean Squared Error */
                                        isRMSE,                                 /* calculate RMSE */
                                        nloops);                                /* number of iteration for each number of clusters */

        //cluster_timing = omp_get_wtime() - cluster_timing;


        /* =============== Command Line Output =============== */

        /* cluster center coordinates
           :displayed only for when k=1*/
        if((min_nclusters == max_nclusters) && (isOutput == 1)) {
                printf("\n================= Centroid Coordinates =================\n");
                for(i = 0; i < max_nclusters; i++){
                        printf("%d:", i);
                        for(j = 0; j < nfeatures; j++){
                                printf(" %.2f", cluster_centres[i][j]);
                        }
                        printf("\n\n");
                }
        }

        len = (float) ((max_nclusters - min_nclusters + 1)*nloops);

        printf("Number of Iteration: %d\n", nloops);
        //printf("Time for I/O: %.5fsec\n", io_timing);
        //printf("Time for Entire Clustering: %.5fsec\n", cluster_timing);

        if(min_nclusters != max_nclusters){
                if(nloops != 1){                                                                        //range of k, multiple iteration
                        //printf("Average Clustering Time: %fsec\n",
                        //              cluster_timing / len);
                        printf("Best number of clusters is %d\n", best_nclusters);
                }
                else{                                                                                           //range of k, single iteration
                        //printf("Average Clustering Time: %fsec\n",
                        //              cluster_timing / len);
                        printf("Best number of clusters is %d\n", best_nclusters);
                }
        }
        else{
                if(nloops != 1){                                                                        // single k, multiple iteration
                        //printf("Average Clustering Time: %.5fsec\n",
                        //              cluster_timing / nloops);
                        if(isRMSE)                                                                              // if calculated RMSE
                                printf("Number of trials to approach the best RMSE of %.3f is %d\n", rmse, index + 1);
                }
                else{                                                                                           // single k, single iteration                           
                        if(isRMSE)                                                                              // if calculated RMSE
                                printf("Root Mean Squared Error: %.3f\n", rmse);
                }
        }

        /* free up memory */
        free(features[0]);
        free(features);
        //MPI_CHECK(MPI_Finalize());
    return(0);
}


////////////////////////////////////////////////////////////////////////////////
// Program main                                                                                                                           //

int
//main( int argc, char** argv) 
kmeans_wrap_main( int argc, char** argv)
{
        // make sure we're running on the big card
    //cudaSetDevice(1);
        // as done in the CUDA start/help document provided
        setup(argc, argv);
        return 0;
}

//                                                                                                                                                        //
////////////////////////////////////////////////////////////////////////////////
#endif

