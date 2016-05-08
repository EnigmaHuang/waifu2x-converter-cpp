#define BLOCK_LOW(id,p,n) ( (id)*(n) / (p) )
#define BLOCK_HIGH(id,p,n) ( BLOCK_LOW((id)+1,p,n) - 1 )
#define BLOCK_SIZE(id,p,n) ( BLOCK_HIGH(id,p,n) - BLOCK_LOW(id,p,n) + 1 )

#include "myConvKernel.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>

#define VEC_WIDTH 8

float *weights, *packed_weights, *inputPlanes, *outputPlanes;
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

void copyFromMemMatF(float *src, float *dst, const int nRow, const int nCol, const int lds, const int ldd)
{
    for (int i = 0; i < nRow; i++)
    {
        float *src_base = src + i * lds;
        float *dst_base = dst + i * ldd;
        memcpy(dst_base, src_base, sizeof(float) * nCol);
    }
}

void padBorderWith3x3Kernel(float *inputPlane)
{
    float *inputPlaneLeftTop = inputPlane + paddedInWidth + 1;
    float *inputPlaneLeftButtom = inputPlaneLeftTop + paddedInWidth * (ioHeight - 1);
    float *ptr_IPTop = inputPlane + 1;
    float *ptr_IPButtom = inputPlane + paddedInWidth * (paddedInHeight - 1) + 1;
    
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

void padInputPlaneCVWith3x3Kernel(cv::Mat _inputPlane, float *inputPlane)
{
    // Copy the same matrix in the middle
    float *inputPlaneLeftTop = inputPlane + paddedInWidth + 1;
    copyFromCVMatF(_inputPlane, inputPlaneLeftTop, ioHeight, ioWidth, paddedInWidth);
    // Padding the border
    padBorderWith3x3Kernel(inputPlane);
}

void padInputPlaneWith3x3KernelFromOutput(float *_outputPlane, float *inputPlane)
{
    // Copy the same matrix in the middle
    float *inputPlaneLeftTop = inputPlane + paddedInWidth + 1;
    copyFromMemMatF(_outputPlane, inputPlaneLeftTop, ioHeight, ioWidth, ioWidth, paddedInWidth);
    padBorderWith3x3Kernel(inputPlane);
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
    
    weights        = (float*)  _mm_malloc(sizeof(float)  * wSize        * max_nWeights,      512);
    packed_weights = (float*)  _mm_malloc(sizeof(float)  * wSize        * max_nWeights,      512);
    inputPlanes    = (float*)  _mm_malloc(sizeof(float)  * paddedInSize * max_nInputPlanes,  512);
    outputPlanes   = (float*)  _mm_malloc(sizeof(float)  * outputSize   * max_nOutputPlanes, 512);
    biases         = (double*) _mm_malloc(sizeof(double) * max_nWeights,                     512);
    assert(weights != NULL && packed_weights != NULL && inputPlanes != NULL 
           && outputPlanes != NULL && biases != NULL);
    
    // outputPlanes are all zero matrices, though it should be reset before each time
    memset(outputPlanes, 0, sizeof(float) * outputSize * max_nOutputPlanes);
    
    // The first input plane is THE ONLY ONE input, others are from output planes
    // To make the operation same, copy this input plane to the 1st output plane,
    // it will be copied to the input plane in the 1st round
    copyFromCVMatF(_1stInputPlane, outputPlanes, ioHeight, ioWidth, ioWidth);
}

void repack3x3Kernels(float *ori_weights, float *packed_weights)
{
    for (int ii = 0; ii < nInputPlanes; ii++)
    {
        for (int oi = 0; oi < nOutputPlanes; oi++)
        {
            int wIndex = oi * nInputPlanes + ii;
            int packed_wIndex = ii * nOutputPlanes + oi;
            float *ori_wi = ori_weights + wIndex * 9;
            for (int k = 0; k < 9; k++)
                packed_weights[k * nWeights + packed_wIndex] = ori_wi[k];
        }
    }
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
    repack3x3Kernels(weights, packed_weights);
    
    // copy baises to local
    for (int i = 0; i < nWeights; i++)
        biases[i] = _biases[i];
}

void convolve3x3withPad(
    float *inputPlane, float *outputPlane, float *weightMatrix,
    const int ioHeight_spos, const int ioHeight_epos
)
{   
    int paddedInWidth = ioWidth + 2;
    for (int opY = ioHeight_spos; opY < ioHeight_epos; opY++)
    {
        for (int opX = 0; opX < ioWidth; opX++)
        {
            register float res = 0.0;
            res += inputPlane[(opY    ) * paddedInWidth + (opX    )] * weightMatrix[0];
            res += inputPlane[(opY    ) * paddedInWidth + (opX + 1)] * weightMatrix[1];
            res += inputPlane[(opY    ) * paddedInWidth + (opX + 2)] * weightMatrix[2];
            res += inputPlane[(opY + 1) * paddedInWidth + (opX    )] * weightMatrix[3];
            res += inputPlane[(opY + 1) * paddedInWidth + (opX + 1)] * weightMatrix[4];
            res += inputPlane[(opY + 1) * paddedInWidth + (opX + 2)] * weightMatrix[5];
            res += inputPlane[(opY + 2) * paddedInWidth + (opX    )] * weightMatrix[6];
            res += inputPlane[(opY + 2) * paddedInWidth + (opX + 1)] * weightMatrix[7];
            res += inputPlane[(opY + 2) * paddedInWidth + (opX + 2)] * weightMatrix[8];
            outputPlane[opY  * ioWidth + opX] = res;
        }
    } 
}

void convolve3x3withPad_1line(
    float *inputPlane, float *outputPlane, float *weightMatrix,
    const int ioHeight_spos, const int ioHeight_epos
)
{   
    int paddedInWidth = ioWidth + 2;
    for (int opY = ioHeight_spos; opY < ioHeight_epos; opY++)
    {
        float *oP_base = outputPlane + opY * ioWidth;
        memset(oP_base, 0, sizeof(float) * ioWidth);
        
        for (int shiftY = 0; shiftY < 3; shiftY++)
            for (int shiftX = 0; shiftX < 3; shiftX++)
            {
                float *iP_spos = inputPlane + (opY + shiftY) * paddedInWidth + shiftX;
                float w = weightMatrix[shiftY * 3 + shiftX];
                
                #pragma simd
                for (int opX = 0; opX < ioWidth; opX++)
                    oP_base[opX] += w * iP_spos[opX];
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

void myConvKernel_naive()
{
    float *filterOutput_buf = (float*) _mm_malloc(sizeof(float) * outputSize, 512); 
    assert(filterOutput_buf != NULL);
    
    memset(outputPlanes, 0, outputSize * nOutputPlanes);
    
    #pragma omp parallel
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
                    ioHeight_spos, ioHeight_epos
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

void convolve3x3withPad_1elem(const int opY, const int opX, float *intermediate)
{
    memset(intermediate, 0, sizeof(float) * nOutputPlanes);
    float input3x3[9];
    for (int ipIndex = 0; ipIndex < nInputPlanes; ipIndex++)
    {
        float *inputPlane = inputPlanes + paddedInSize * ipIndex;
        input3x3[0] = inputPlane[(opY + 0) * paddedInWidth + (opX + 0)];
        input3x3[1] = inputPlane[(opY + 0) * paddedInWidth + (opX + 1)];
        input3x3[2] = inputPlane[(opY + 0) * paddedInWidth + (opX + 2)];
        input3x3[3] = inputPlane[(opY + 1) * paddedInWidth + (opX + 0)];
        input3x3[4] = inputPlane[(opY + 1) * paddedInWidth + (opX + 1)];
        input3x3[5] = inputPlane[(opY + 1) * paddedInWidth + (opX + 2)];
        input3x3[6] = inputPlane[(opY + 2) * paddedInWidth + (opX + 0)];
        input3x3[7] = inputPlane[(opY + 2) * paddedInWidth + (opX + 1)];
        input3x3[8] = inputPlane[(opY + 2) * paddedInWidth + (opX + 2)];
        float *w_base = packed_weights + ipIndex * nOutputPlanes;
        for (int k = 0; k < 9; k++)
        {
            float *w_spos = w_base + k * nWeights;
            
            #pragma simd
            for (int i = 0; i < nOutputPlanes; i++)
                intermediate[i] += w_spos[i] * input3x3[k];
        }
    }
    
    for (int opIndex = 0; opIndex < nOutputPlanes; opIndex++)
    {
        intermediate[opIndex] += biases[opIndex];
        if (intermediate[opIndex] < 0) 
            intermediate[opIndex] *= 0.1;
        outputPlanes[opIndex * outputSize + opY * ioWidth + opX] = intermediate[opIndex];
    }
}

void myConvKernel_simd()
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        float *intermediate = (float*) _mm_malloc(sizeof(float) * nOutputPlanes, 512);
        assert(intermediate != NULL);
        
        #pragma omp for
        for (int opY = 0; opY < ioHeight; opY++)
        {
            for (int opX = 0; opX < ioWidth; opX++)
            {
                convolve3x3withPad_1elem(opY, opX, intermediate);
            }
        }
        
        _mm_free(intermediate);
    }
}

void myConvKernel()
{
    //if (nOutputPlanes % VEC_WIDTH)
    //{
        myConvKernel_naive();
    //} else { 
    //   myConvKernel_simd();
    //}
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
    
    if (weights != NULL)        _mm_free(weights);
    if (packed_weights != NULL) _mm_free(packed_weights);
    if (inputPlanes != NULL)    _mm_free(inputPlanes);
    if (outputPlanes != NULL)   _mm_free(outputPlanes);
    if (biases != NULL)         _mm_free(biases);
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
    printf("Total computing time = %lf (seconds)\n", totalTimeCost);
    printf("Total GFlops         = %lf (single precision)\n", res);
    printf("====================================\n");
}