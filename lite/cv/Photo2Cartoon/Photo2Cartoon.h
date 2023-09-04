//
// Created by wangzijian on 2023/8/23.
//

#include "string"
#include "onnxruntime_cxx_api.h"
#include "iostream"
#include "vector"
#include "opencv2/opencv.hpp"
#include "time.h"

using namespace std;
using namespace Ort;
using namespace cv;

class Photo2Cartoon {
public:
    void preprocessMat(Mat inputMat, Mat &normalized);

    void postprocessMat(const float *outputDataPtr, int resize_w, int resize_h, Mat &outPutMat);


    std::shared_ptr<float> runNet(cv::Mat normalized);

    void detect1(string srcImg, string outImg);

    void detect(string srcImg, string outImg);

private:
#ifdef linux
    Ort::Env env;
    Ort::Session session_{nullptr};
    Ort::Session session_cartooin{nullptr};

    // 这里必须要更正一下linux和windows的加载路径的方法有些不同

#elif _WIN32
    Ort::Session session_{env, L"E:\\lite.ai.toolkit.win\\models\\lite\\cv\\Photo2Cartoon\\minivision_head_seg.onnx",
                          SessionOptions{nullptr}};


    Ort::Session session_cartooin{env,
                                  L"E:\\lite.ai.toolkit.win\\models\\lite\\cv\\Photo2Cartoon\\Photo2Cartoon_fp32.onnx",
                                  SessionOptions{nullptr}};
#endif



};


