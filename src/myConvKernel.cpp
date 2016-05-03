#include "myConvKernel.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

float *weights, *inputPlanes, *outputPlanes;
double *biases;
int nInputPlanes, nOutputPlanes, nWeights;
int wWidth, wHeight, ioWidth, ioHeight, paddedInWidth, paddedInHeight;
int wSize, paddedInSize, outputSize;

void copyFromCVMatF(const cv::Mat &src, float *dst, const int nRow, const int nCol, const int ldd)
{
    for (int i = 0; i < nRow; i++)
    {
        float *dst_base = dst + i * ldd;
        const float *src_base = src.ptr<float>(i);
        memcpy(dst_base, src_base, sizeof(float) * nCol);
    }
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

void padInputPlaneWith3x3Kernel(cv::Mat _inputPlane, float *inputPlane)
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

void copyInMatrices(
    const int _nInputPlanes, const int _nOutputPlanes,
    const int _wWidth, const int _wHeight,   const std::vector<cv::Mat> &_weights,
    const int _ioWidth, const int _ioHeight, const std::vector<cv::Mat> &_inputPlanes,
    const std::vector<double> _biases
)
{
    nInputPlanes   = _nInputPlanes;
    nOutputPlanes  = _nOutputPlanes;
    nWeights       = nInputPlanes * nOutputPlanes;
    
    wWidth         = _wWidth;    // This should be 3
    wHeight        = _wHeight;   // This should be 3, too
    ioWidth        = _ioWidth;
    ioHeight       = _ioHeight;
    paddedInWidth  = ioWidth + wWidth - 1;
    paddedInHeight = ioHeight + wHeight - 1;
    
    wSize          = wWidth * wHeight;
    paddedInSize   = paddedInWidth * paddedInHeight;
    outputSize     = ioWidth * ioHeight;
    
    weights      = (float*)  _mm_malloc(sizeof(float)  * wSize        * nWeights,      512);
    inputPlanes  = (float*)  _mm_malloc(sizeof(float)  * paddedInSize * nInputPlanes,  512);
    outputPlanes = (float*)  _mm_malloc(sizeof(float)  * outputSize   * nOutputPlanes, 512);
    biases       = (double*) _mm_malloc(sizeof(double) * nWeights,                     512);
    assert(weights != NULL && inputPlanes != NULL && outputPlanes != NULL && biases != NULL);
    
    // copy baises to local
    for (int i = 0; i < nWeights; i++)
        biases[i] = _biases[i];
    
    // outputPlanes are all zero matrices
    memset(outputPlanes, 0, sizeof(float) * outputSize * nOutputPlanes);
    
    // pad for input planes to simplify the compute kernel
    memset(inputPlanes, 0, sizeof(float) * paddedInSize * nInputPlanes);
    for (int i = 0; i < nInputPlanes; i++)
        padInputPlaneWith3x3Kernel(_inputPlanes[i], inputPlanes + paddedInSize * i);
    
    // copy weightMatrices to local
    for (int i = 0; i < nWeights; i++)
        copyFromCVMatF(_weights[i], weights + wSize * i, wHeight, wWidth, wWidth);
}

void convolve3x3withPad(
    float *inputPlane, float *outputPlane, float *weightMatrix,
    const int ioWidth, const int ioHeight, const int wWidth, const int wHeight,
    const int paddedInWidth, const int paddedInHeight
)
{   
    for (int ipY = 1; ipY < ioHeight + 1; ipY++)
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
    #pragma simd
    for (int i = 0; i < length; i++) dst[i] += src[i];
}

void addScale(const int length, const float delta, float *dst)
{
    #pragma simd 
    for (int i = 0; i < length; i++) dst[i] += delta;
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
    
    int semi_wWidth = wWidth / 2;
    
    #pragma omp parallel for
    for (int opIndex = 0; opIndex < nOutputPlanes; opIndex++)
    {
        int tid = omp_get_thread_num();
        float *filterOutput = filterOutput_buf + outputSize * tid;    // filterOutput in modelHandler.cpp
        
        int wMatIndex = nInputPlanes * opIndex;
        float *outputPlane = outputPlanes + opIndex * outputSize;     // uIntermediatePlane & outputPlane in modelHandler.cpp
        memset(outputPlane, 0, sizeof(float) * outputSize);
        
        for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++) 
        {
            float *inputPlane = inputPlanes + ipIndex * paddedInSize; // uInputPlane  in modelHandler.cpp
            float *weightMatrix = weights + (wMatIndex++) * wSize;    // weightMatrix in modelHandler.cpp
            
            convolve3x3withPad(
                inputPlane, filterOutput, weightMatrix,
                ioWidth, ioHeight, wWidth, wHeight, 
                paddedInWidth, paddedInHeight
            );
            
            addVec(outputSize, filterOutput, outputPlane);            // cv::add(uIntermediatePlane, filterOutput, uIntermediatePlane)
        }
        
        addScale(outputSize, (float)(biases[opIndex]), outputPlane);  // cv::add(uIntermediatePlane, biases[opIndex], uIntermediatePlane);
        scaleIfLessThanX(outputSize, outputPlane, 0.0, 0.1);          // cv::scaleAdd(lessThanZero, 0.1, moreThanZero, uIntermediatePlane);
    }

    _mm_free(filterOutput_buf);
}

void copyOutResults(std::vector<cv::Mat> &_outputPlanes)
{   
    for (int i = 0; i < nOutputPlanes; i++)
        copyToCVMatF(outputPlanes + outputSize * i, _outputPlanes[i], ioHeight, ioWidth, ioWidth);
    
    _mm_free(weights);
    _mm_free(inputPlanes);
    _mm_free(outputPlanes);
    _mm_free(biases);
}