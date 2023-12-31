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
    CodeFormer();
    void preprocess(cv::Mat &input);
    void postprocess(float *pred,string outImg);
    float* runNet(cv::Mat input);
    void detect(string srcImg,string outImg);
    void detect1(string srcImg,string outImg);




private:
    Ort::Env env;
    Ort::Session session_{env,
                          "/home/wangzijian/Desktop/lite.ai.toolkit.win/models/lite/cv/Resolution/codeformer.onnx",
                          Ort::SessionOptions{nullptr}};

    vector<char*> input_names;
    vector<char*> output_names;
    // 定义onnx所需要的输入维度
    int input_channels;
    int input_height;
    int input_width;

    // 定义原图的宽和高
    int ori_h;
    int ori_w;

    vector<vector<int64_t>> input_node_dims; // >=1 outputs
    vector<vector<int64_t>> output_node_dims; //
    float min_max[2] = { -1,1 };


};

