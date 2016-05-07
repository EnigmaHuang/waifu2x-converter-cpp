#define BLOCK_LOW(id,p,n) ( (id)*(n) / (p) )
#define BLOCK_HIGH(id,p,n) ( BLOCK_LOW((id)+1,p,n) - 1 )
#define BLOCK_SIZE(id,p,n) ( BLOCK_HIGH(id,p,n) - BLOCK_LOW(id,p,n) + 1 )

#include "myConvKernel.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

float *weights, *inputPlanes, *outputPlanes;
double *biases;
int max_nInputPlanes, max_nOutputPlanes, max_nWeights;
int nInputPlanes, nOutputPlanes, nWeights;
int wWidth, wHeight, ioWidth, ioHeight, paddedInWidth, paddedInHeight;
int wSize, paddedInSize, outputSize;

double totalGflops, totalTimeCost;

void copyFromCVMatF(const cv::Mat &src, float *dst, const int nRow, const int nCol, const int ldd)
{
    for (int i = 0; i < nRow; i++)
    {
        float *dst_base = dst + i * ldd;
        const float *src_base = src.ptr<float>(i);
        memcpy(dst_base, src_base, sizeof(float) * nCol);
    }
}

void padInputPlaneCVWith3x3Kernel(cv::Mat _inputPlane, float *inputPlane)
{
    float *inputPlaneLeftTop = inputPlane + paddedInWidth + 1;
    float *inputPlaneLeftButtom = inputPlaneLeftTop + paddedInWidth * (ioHeight - 1);
    float *ptr_IPTop = inputPlane + 1;
    float *ptr_IPButtom = inputPlane + paddedInWidth * (paddedInHeight - 1) + 1;
    
    // Copy the same matrix in the middle
    copyFromCVMatF(_inputPlane, inputPlaneLeftTop, ioHeight, ioWidth, paddedInWidth);
    
    // Fill the top and buttom 
    memcpy(ptr_IPTop, inputPlaneLeftTop, sizeof(float) * ioWidth);
    memcpy(ptr_IPButtom, inputPlaneLeftButtom, sizeof(float) * ioWidth);
    
    // Fill the left and right sides
    float *ptr_IPLeft = inputPlane + paddedInWidth;
    float *ptr_IPRight = ptr_IPLeft + ioWidth;
    for (int iRow = 0; iRow < ioHeight; iRow++)
    {
        ptr_IPLeft[0] = ptr_IPLeft[1];
        ptr_IPRight[1] = ptr_IPRight[0];
        ptr_IPLeft += paddedInWidth;
        ptr_IPRight += paddedInWidth;
    }
    
    // Copy 4 elements on the corner
    inputPlane[0] = inputPlane[paddedInWidth + 1];
    inputPlane[paddedInWidth - 1] = inputPlane[2 * paddedInWidth - 2];
    inputPlane[paddedInWidth * (paddedInHeight - 1)] = inputPlane[paddedInWidth * (paddedInHeight - 2) + 1];
    inputPlane[paddedInWidth * paddedInHeight - 1] = inputPlane[paddedInWidth * (paddedInHeight - 1) - 2];
}

void copyFromMemMatF(float *src, float *dst, const int nRow, const int nCol, const int lds, const int ldd)
{
    for (int i = 0; i < nRow; i++)
    {
        float *src_base = src + i * lds;
        float *dst_base = dst + i * ldd;
        memcpy(dst_base, src_base, sizeof(float) * nCol);
    }
}

void padInputPlaneWith3x3KernelFromOutput(float *_outputPlane, float *inputPlane)
{
    float *inputPlaneLeftTop = inputPlane + paddedInWidth + 1;
    float *inputPlaneLeftButtom = inputPlaneLeftTop + paddedInWidth * (ioHeight - 1);
    float *ptr_IPTop = inputPlane + 1;
    float *ptr_IPButtom = inputPlane + paddedInWidth * (paddedInHeight - 1) + 1;
    
    // Copy the same matrix in the middle
    copyFromMemMatF(_outputPlane, inputPlaneLeftTop, ioHeight, ioWidth, ioWidth, paddedInWidth);
    
    // Fill the top and buttom 
    memcpy(ptr_IPTop, inputPlaneLeftTop, sizeof(float) * ioWidth);
    memcpy(ptr_IPButtom, inputPlaneLeftButtom, sizeof(float) * ioWidth);
    
    // Fill the left and right sides
    float *ptr_IPLeft = inputPlane + paddedInWidth;
    float *ptr_IPRight = ptr_IPLeft + ioWidth;
    for (int iRow = 0; iRow < ioHeight; iRow++)
    {
        ptr_IPLeft[0] = ptr_IPLeft[1];
        ptr_IPRight[1] = ptr_IPRight[0];
        ptr_IPLeft += paddedInWidth;
        ptr_IPRight += paddedInWidth;
    }
    
    // Copy 4 elements on the corner
    inputPlane[0] = inputPlane[paddedInWidth + 1];
    inputPlane[paddedInWidth - 1] = inputPlane[2 * paddedInWidth - 2];
    inputPlane[paddedInWidth * (paddedInHeight - 1)] = inputPlane[paddedInWidth * (paddedInHeight - 2) + 1];
    inputPlane[paddedInWidth * paddedInHeight - 1] = inputPlane[paddedInWidth * (paddedInHeight - 1) - 2];
}

void initLocalMem(
    const int _max_nInputPlanes, const int _max_nOutputPlanes,
    const int _ioWidth, const int _ioHeight, cv::Mat _1stInputPlane,
    const int _wWidth, const int _wHeight
)
{
    max_nInputPlanes  = _max_nInputPlanes;
    max_nOutputPlanes = _max_nOutputPlanes;
    max_nWeights      = max_nInputPlanes * max_nOutputPlanes;
    
    wWidth         = _wWidth;    // This should be 3
    wHeight        = _wHeight;   // This should be 3, too
    ioWidth        = _ioWidth;
    ioHeight       = _ioHeight;
    paddedInWidth  = ioWidth + wWidth - 1;
    paddedInHeight = ioHeight + wHeight - 1;
    wSize          = wWidth * wHeight;
    paddedInSize   = paddedInWidth * paddedInHeight;
    outputSize     = ioWidth * ioHeight;
    
    weights      = (float*)  _mm_malloc(sizeof(float)  * wSize        * max_nWeights,      512);
    inputPlanes  = (float*)  _mm_malloc(sizeof(float)  * paddedInSize * max_nInputPlanes,  512);
    outputPlanes = (float*)  _mm_malloc(sizeof(float)  * outputSize   * max_nOutputPlanes, 512);
    biases       = (double*) _mm_malloc(sizeof(double) * max_nWeights,                     512);
    assert(weights != NULL && inputPlanes != NULL && outputPlanes != NULL && biases != NULL);
    
    // outputPlanes are all zero matrices, though it should be reset before each time
    memset(outputPlanes, 0, sizeof(float) * outputSize * max_nOutputPlanes);
    
    // The first input plane is THE ONLY ONE input, others are from output planes
    // To make the operation same, copy this input plane to the 1st output plane,
    // it will be copied to the input plane in the 1st round
    copyFromCVMatF(_1stInputPlane, outputPlanes, ioHeight, ioWidth, ioWidth);
}

void copyInMatrices(
    const int _nInputPlanes, const int _nOutputPlanes,
    const std::vector<cv::Mat> &_weights, const std::vector<double> _biases
)
{
    nInputPlanes   = _nInputPlanes;
    nOutputPlanes  = _nOutputPlanes;
    nWeights       = nInputPlanes * nOutputPlanes;
    
    // pad for input planes to simplify the compute kernel
    // inputPlanes are the outputPlanes from the previous round
    memset(inputPlanes, 0, sizeof(float) * paddedInSize * nInputPlanes);
    for (int i = 0; i < nInputPlanes; i++)
        padInputPlaneWith3x3KernelFromOutput(outputPlanes + outputSize * i, inputPlanes + paddedInSize * i);
    
    // outputPlanes are all zero matrices
    memset(outputPlanes, 0, sizeof(float) * outputSize * nOutputPlanes);
    
    // copy weightMatrices to local
    for (int i = 0; i < nWeights; i++)
        copyFromCVMatF(_weights[i], weights + wSize * i, wHeight, wWidth, wWidth);
    
    // copy baises to local
    for (int i = 0; i < nWeights; i++)
        biases[i] = _biases[i];
}

void convolve3x3withPad(
    float *inputPlane, float *outputPlane, float *weightMatrix,
    const int ioWidth, const int ioHeight, const int ioHeight_spos, const int ioHeight_epos
)
{   
    int paddedInWidth = ioWidth + 2;
    for (int ipY = 1 + ioHeight_spos; ipY < 1 + ioHeight_epos; ipY++)
    {
        for (int ipX = 1; ipX < ioWidth + 1; ipX++)
        {
            register float res = 0.0;
            res += inputPlane[(ipY - 1) * paddedInWidth + (ipX - 1)] * weightMatrix[0];
            res += inputPlane[(ipY - 1) * paddedInWidth + (ipX    )] * weightMatrix[1];
            res += inputPlane[(ipY - 1) * paddedInWidth + (ipX + 1)] * weightMatrix[2];
            res += inputPlane[(ipY    ) * paddedInWidth + (ipX - 1)] * weightMatrix[3];
            res += inputPlane[(ipY    ) * paddedInWidth + (ipX    )] * weightMatrix[4];
            res += inputPlane[(ipY    ) * paddedInWidth + (ipX + 1)] * weightMatrix[5];
            res += inputPlane[(ipY + 1) * paddedInWidth + (ipX - 1)] * weightMatrix[6];
            res += inputPlane[(ipY + 1) * paddedInWidth + (ipX    )] * weightMatrix[7];
            res += inputPlane[(ipY + 1) * paddedInWidth + (ipX + 1)] * weightMatrix[8];
            outputPlane[(ipY - 1) * ioWidth + (ipX - 1)] = res;
        }
    }   
}

void addVec(const int length, float *src, float *dst)
{
    for (int i = 0; i < length; i++) dst[i] += src[i];
}

void addBias(const int length, const float bias, float *dst)
{
    for (int i = 0; i < length; i++) dst[i] += bias;
}

void scaleIfLessThanX(const int length, float *dst, const float X, const float alpha)
{
    for (int i = 0; i < length; i++)
        if (dst[i] < X)
            dst[i] *= alpha;
}

void myConvKernel()
{
    int nCPUThreads;
    #pragma omp parallel
    {
        #pragma omp master 
        nCPUThreads = omp_get_num_threads();
    }
    
    float *filterOutput_buf = (float*) _mm_malloc(sizeof(float) * outputSize * nCPUThreads, 512); 
    assert(filterOutput_buf != NULL);
    
    int ipIndexStep = 8;
    
    memset(outputPlanes, 0, outputSize * nOutputPlanes);
    
    #pragma omp parallel num_threads(nCPUThreads)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        int ioHeight_spos = BLOCK_LOW(tid, nthreads, ioHeight);
        int ioHeight_epos = BLOCK_LOW(tid + 1, nthreads, ioHeight);
        
        int oS_spos = ioHeight_spos * ioWidth;
        int oS_size = (ioHeight_epos - ioHeight_spos) * ioWidth;
      
        for (int opIndex = 0; opIndex < nOutputPlanes; opIndex++)
        {
            float *filterOutput = filterOutput_buf;                    
            float *outputPlane = outputPlanes + opIndex * outputSize; 
            
            for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++)
            {
                int wMatIndex = nInputPlanes * opIndex + ipIndex;
                float *inputPlane = inputPlanes + ipIndex * paddedInSize;
                float *weightMatrix = weights + wMatIndex * wSize;
                
                convolve3x3withPad(
                    inputPlane, filterOutput, weightMatrix,
                    ioWidth, ioHeight, ioHeight_spos, ioHeight_epos
                );

                addVec(oS_size, filterOutput + oS_spos, outputPlane + oS_spos);
            }
        }
        
        #pragma omp barrier
        
        #pragma omp for
        for (int opIndex = 0; opIndex < nOutputPlanes; opIndex++)
        {
            int wMatIndex = nInputPlanes * opIndex;
            float *outputPlane = outputPlanes + opIndex * outputSize;    
            addBias(outputSize, (float)(biases[opIndex]), outputPlane); 
            scaleIfLessThanX(outputSize, outputPlane, 0.0, 0.1);  
        }
    }

    _mm_free(filterOutput_buf);
}

void copyToCVMatF(const float *src, cv::Mat &dst, const int nRow, const int nCol, const int lds)
{
    for (int i = 0; i < nRow; i++)
    {
        const float *src_base = src + i * lds;
        float *dst_base = dst.ptr<float>(i);
        memcpy(dst_base, src_base, sizeof(float) * nCol);
    }
}

void copyOutResults(std::vector<cv::Mat> &_outputPlanes)
{   
    copyToCVMatF(outputPlanes, _outputPlanes[0], ioHeight, ioWidth, ioWidth);
    
    if (weights != NULL)      _mm_free(weights);
    if (inputPlanes != NULL)  _mm_free(inputPlanes);
    if (outputPlanes != NULL) _mm_free(outputPlanes);
    if (biases != NULL)       _mm_free(biases);
}

void resetTotalGFlops()
{
    totalGflops   = 0.0;
    totalTimeCost = 0.0;
}

void addGFlops(double newGFlops, double newTimeCost)
{
    totalGflops   += newGFlops;
    totalTimeCost += newTimeCost;
}

void reportTotalGFlops()
{
    double res = totalGflops / totalTimeCost;
    printf("\n===== Total Performance Report =====\n");
    printf("Total time   = %lf (seconds)\n", totalTimeCost);
    printf("Total GFlops = %lf (single precision)\n", res);
    printf("====================================\n");
}