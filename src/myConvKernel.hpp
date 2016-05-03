#ifndef MY_CONV_KERNEL_H
#define MY_CONV_KERNEL_H

#include <opencv2/opencv.hpp>

void copyInMatrices(
    const int _nInputPlanes, const int _nOutputPlanes,
    const int _wWidth, const int _wHeight,   const std::vector<cv::Mat> &_weights,
    const int _ioWidth, const int _ioHeight, const std::vector<cv::Mat> &_inputPlanes,
    const std::vector<double> _biases
);

void myConvKernel();

void copyOutResults(std::vector<cv::Mat> &_outputPlanes);

#endif