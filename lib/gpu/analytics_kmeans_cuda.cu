#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <omp.h>

#include <cuda.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM

#include "analytics_kmeans_cuda_cu.h"
#include "analytics_kmeans_cuda_kernel.cu"

//#define BLOCK_DELTA_REDUCE
//#define BLOCK_CENTER_REDUCE

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

// GLOBAL!!!!!
unsigned int num_threads_perdim = THREADS_PER_DIM;					/* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM;					/* temporary */
unsigned int num_threads = num_threads_perdim*num_threads_perdim;	/* number of threads */
unsigned int num_blocks = num_blocks_perdim*num_blocks_perdim;		/* number of blocks */

/* _d denotes it resides on the device */
int    *membership_new;												/* newly assignment membership */
float  *feature_d;													/* inverted data array */
float  *feature_flipped_d;											/* original (not inverted) data array */
int    *membership_d;												/* membership on the device */
float  *block_new_centers;											/* sum of points in a cluster (per block) */
float  *clusters_d;													/* cluster centers on the device */
float  *block_clusters_d;											/* per block calculation of cluster centers */
int    *block_deltas_d;												/* per block calculation of deltas */


/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
extern "C"
void allocateMemory(int npoints, int nfeatures, int nclusters, float **features)
{	
	num_blocks = npoints / num_threads;
	if (npoints % num_threads > 0)		/* defeat truncation */
		num_blocks++;

	num_blocks_perdim = sqrt((double) num_blocks);
	while (num_blocks_perdim * num_blocks_perdim < num_blocks)	// defeat truncation (should run once)
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
int	// delta -- had problems when return value was of float type
kmeansCuda(float  **feature,				/* in: [npoints][nfeatures] */
           int      nfeatures,				/* number of attributes for each point */
           int      npoints,				/* number of data points */
           int      nclusters,				/* number of clusters */
           int     *membership,				/* which cluster the point belongs to */
		   float  **clusters,				/* coordinates of cluster centers */
		   int     *new_centers_len,		/* number of elements in each cluster */
           float  **new_centers				/* sum of elements in each cluster */
		   )
{
	int delta = 0;			/* if point has moved */
	int i,j;				/* counters */


	//cudaSetDevice(1);

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
			if(new_centers[j][k] >	1.001 * block_new_centers[j*nfeatures + k] || new_centers[j][k] <	0.999 * block_new_centers[j*nfeatures + k]) {
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
        //printf("iterated %d times\n", c);
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
                //int             nloops = 1;                             /* default value */
                int             nloops = 1000;                             /* default value */

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
// Program main																  //

int
//main( int argc, char** argv) 
kmeans_wrap_main( int argc, char** argv) 
{
	// make sure we're running on the big card
    //cudaSetDevice(1);
	// as done in the CUDA start/help document provided
        cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
	{
		printf("compute mode %d\n", (int)prop.computeMode);
	}
	setup(argc, argv);    
	return 0;
}

//																			  //
////////////////////////////////////////////////////////////////////////////////



