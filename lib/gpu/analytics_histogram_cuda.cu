/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
* This sample implements 64-bin histogram calculation
* of arbitrary-sized 8-bit data array
*/

// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include "helper_cuda.h"
#include "helper_functions.h"  // helper for shared that are common to CUDA SDK samples

// project include
#include "histogram_common.h"
#include "analytics_histogram_cuda_cu.h"

#include <sys/types.h>
#include <unistd.h>

const int numRuns = 16;
static char *sSDKsample = "[histogram]\0";

int histogram_wrap_main(int argc, char **argv)
{
    uchar *h_Data;
    uint  *h_HistogramCPU, *h_HistogramGPU;
    uchar *d_Data;
    uint  *d_Histogram;
    StopWatchInterface *hTimer = NULL;
    int PassFailFlag = 1;
    uint byteCount = 64 * 1048576;
    uint uiSizeMult = 1;

    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;

    // set logfile name and start logs
    fprintf(stderr,"%d : [%s] - Starting...\n", getpid(), sSDKsample);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //int dev = findCudaDevice(argc, (const char **)argv);

    //checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

    fprintf(stderr,"%d : CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
           getpid(), deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = deviceProp.major * 0x10 + deviceProp.minor;

    if (version < 0x11)
    {
        fprintf(stderr,"There is no device supporting a minimum of CUDA compute capability 1.1 for this SDK sample\n");
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    //sdkCreateTimer(&hTimer);

    // Optional Command-line multiplier to increase size of array to histogram
    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
        uiSizeMult = MAX(1,MIN(uiSizeMult, 10));
        byteCount *= uiSizeMult;
    }

    fprintf(stderr,"%d : Initializing data...\n", getpid());
    fprintf(stderr,"%d : ...allocating CPU memory.\n", getpid());
    h_Data         = (uchar *)malloc(byteCount);
    h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
    h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

    fprintf(stderr,"%d : ...generating input data\n", getpid());
    srand(2009);

    for (uint i = 0; i < byteCount; i++)
    {
        h_Data[i] = rand() % 256;
    }

    fprintf(stderr,"%d : ...allocating GPU memory and copying input data\n\n", getpid());
    checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
    checkCudaErrors(cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));

#define COUNT 2
for(int i=0; i<1000;/*i<COUNT;*/ i++){

        if(i%10 == 0)
	fprintf(stderr, "HISTOGRAM iteration %d\n", i);

    checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));

    {
        //fprintf(stderr,"%d : Starting up 64-bin histogram...\n\n", getpid());
        initHistogram64();

        //fprintf(stderr,"%d : Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n", getpid(), byteCount, numRuns);

        for (int iter = -1; iter < numRuns; iter++)
        {
            //iter == -1 -- warmup iteration
            if (iter == 0)
            {
                cudaDeviceSynchronize();
                sdkResetTimer(&hTimer);
                sdkStartTimer(&hTimer);
            }

            histogram64(d_Histogram, d_Data, byteCount);
        }

        cudaDeviceSynchronize();
        sdkStopTimer(&hTimer);
        //double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
        //fprintf(stderr,"histogram64() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
        //fprintf(stderr,"histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
          //     (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM64_THREADBLOCK_SIZE);

        //fprintf(stderr,"\nValidating GPU results...\n");
        //fprintf(stderr,"%d :  ...reading back GPU results\n", getpid());
        checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

        /*fprintf(stderr," ...histogram64CPU()\n");
        histogram64CPU(
            h_HistogramCPU,
            h_Data,
            byteCount
        );

        fprintf(stderr," ...comparing the results...\n");

        for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
            if (h_HistogramGPU[i] != h_HistogramCPU[i])
            {
                PassFailFlag = 0;
            }

        fprintf(stderr,PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n");*/

        //fprintf(stderr,"%d : Shutting down 64-bin histogram...\n\n\n", getpid());
        closeHistogram64();
    }

    {
        //fprintf(stderr,"%d : Initializing 256-bin histogram...\n", getpid());
        initHistogram256();

        //fprintf(stderr,"%d : Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", getpid(), byteCount, numRuns);

        for (int iter = -1; iter < numRuns; iter++)
        {
            //iter == -1 -- warmup iteration
            if (iter == 0)
            {
                checkCudaErrors(cudaDeviceSynchronize());
                sdkResetTimer(&hTimer);
                sdkStartTimer(&hTimer);
            }

            histogram256(d_Histogram, d_Data, byteCount);
        }

        cudaDeviceSynchronize();
        sdkStopTimer(&hTimer);
        //double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
        //fprintf(stderr,"histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
        //fprintf(stderr,"histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
          //     (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);

        //fprintf(stderr,"\nValidating GPU results...\n");
        //fprintf(stderr,"%d :  ...reading back GPU results\n", getpid());
        checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

        /*fprintf(stderr," ...histogram256CPU()\n");
        histogram256CPU(
            h_HistogramCPU,
            h_Data,
            byteCount
        );*/

        /*fprintf(stderr," ...comparing the results\n");

        for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
            if (h_HistogramGPU[i] != h_HistogramCPU[i])
            {
                PassFailFlag = 0;
            }

        fprintf(stderr,PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");*/

        //fprintf(stderr,"%d : Shutting down 256-bin histogram...\n\n\n", getpid());
        closeHistogram256();
    }

}

    //fprintf(stderr,"%d : Shutting down...\n", getpid());
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_Histogram));
    checkCudaErrors(cudaFree(d_Data));
    free(h_HistogramGPU);
    free(h_HistogramCPU);
    free(h_Data);

    //cudaDeviceReset();
    //fprintf(stderr,"%s - Test Summary\n", sSDKsample);

    // pass or fail (for both 64 bit and 256 bit histograms)
    /*if (!PassFailFlag)
    {
        fprintf(stderr,"Test failed!\n");
        exit(EXIT_FAILURE);
    }

    fprintf(stderr,"Test passed\n");*/
    //exit(EXIT_SUCCESS);
}
