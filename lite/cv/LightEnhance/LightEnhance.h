//
// Created by wangzijian on 2023/8/22.
//
# pragma once

#include "string"
#include "onnxruntime_cxx_api.h"
#include "iostream"
#include "vector"
#include "opencv2/opencv.hpp"
#include "time.h"

using namespace std;
using namespace Ort;
using namespace cv;


class LightEnhance {
public:
    void preprocessMat(Mat inputMat, vector<float> &inputArray);

    void postprocessMat(const float *outputDataPtr, int resize_w, int resize_h, Mat &outPutMat);

    void runNet(std::vector<float> input, int h, int w, Ort::Value &outputTensor);

    void detect(string srcImg, string outImg);

private:
#ifdef linux
    Ort::Env env;
    Ort::Session session_{env, "/home/wangzijian/Desktop/lite.ai.toolkit.win/models/lite/cv/LightEnhance/LightEnhance_fp32.onnx",
                          Ort::SessionOptions{nullptr}};
    Ort::Value outputTensor{nullptr};

#elif _WIN32
   Ort::Session session_{env, L"E:\\lite.ai.toolkit.win\\models\\lite\\cv\\LightEnhance\\LightEnhance_fp32.onnx",
                          Ort::SessionOptions{nullptr}};

    // 如果在这里定义记得初始化一下他 {nullptr}
    Ort::Value outputTensor{nullptr};
#endif

};


