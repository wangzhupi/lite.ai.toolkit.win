//
// Created by wangzijian on 2023/9/5.
//

// 这个后面继承到一个里面
#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "iostream"
using namespace std;


class DeticDet {

public:
    void preprocess(cv::Mat inputMat);

    void runNet();

    void postprocess();

    void detect(string srcImg);

private:

    float scale;

    int max_size = 800;

};


