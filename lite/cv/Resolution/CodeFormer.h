//
// Created by wangzijian on 2023/9/7.
//

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "string"
#include "iostream"
using namespace std;

class CodeFormer {
public:
    void preprocess();
    void postprocess();
    void runNet();
    void detect(string srcImg,string outImg);

private:
    Ort::Env env;
    Ort::Session session_{env,
                          "/home/wangzijian/Desktop/lite.ai.toolkit.win/models/lite/cv/Resolution/codeformer.onnx",
                          Ort::SessionOptions{nullptr}};

};

