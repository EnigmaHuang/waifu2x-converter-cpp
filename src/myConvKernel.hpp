#ifndef MY_CONV_KERNEL_H
#define MY_CONV_KERNEL_H

#include <opencv2/opencv.hpp>

void initLocalMem(
    const int _max_nInputPlanes, const int _max_nOutputPlanes,
    const int _ioWidth, const int _ioHeight, cv::Mat _1stInputPlane,
    const int _wWidth, const int _wHeight
);

void copyInMatrices(
    const int _nInputPlanes, const int _nOutputPlanes,
    const std::vector<cv::Mat> &_weights, const std::vector<double> _biases
);

void myConvKernel();

void copyOutResults(std::vector<cv::Mat> &_outputPlanes);

void resetTotalGFlops();

void addGFlops(double newGFlops, double newTimeCost);

void reportTotalGFlops();

#endif