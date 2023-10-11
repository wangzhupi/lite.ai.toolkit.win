//
// Created by wangzijian on 2023/9/7.
//
#include "onnxruntime_cxx_api.h"
#include "iostream"
#include "string"
#include "opencv2/opencv.hpp"

class Informative_Draw {
public:
    // 用于初始化模型的选择

    void postprocess(float *out_ptr, std::vector<int> shape, std::string outImg,int ori_h,int ori_w);

    float *runNet(cv::Mat test, std::vector<int> &shape);

    void detect(std::string srcImg, std::string outImg);

private:
    // 记录onnx的输入
    std::vector<int> shape;
    Ort::Env env;
    Ort::Session session_{env,
                          "/home/wangzijian/Desktop/lite.ai.toolkit.win/models/lite/cv/SketchDraw/anime_style_512x512.onnx",
                          Ort::SessionOptions{nullptr}};


};

