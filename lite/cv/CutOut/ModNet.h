//
// Created by wangzijian on 2023/8/22.
//

#pragma once

#include "string"
#include "iostream"
#include "opencv2/opencv.hpp"
#include "vector"
#include "onnxruntime_cxx_api.h"
#include "time.h"

using namespace cv;
using namespace Ort;
using namespace std;

class ModNet {
public:

    void preprocessMat(Mat img, vector<float> &inputArray, int &resize_h, int &resize_w);

    void get_scale_factor(std::vector<float> &scale, int im_h, int im_w, int ref_size);

    void postprocessMat(const float *outputDataPtr, int resize_w, int resize_h, Mat &outPutMat);

    void runNet(std::vector<float> input, int h, int w, Ort::Value &outputTensor);

    void detect(string srcImg, string outImg);


private:
#ifdef linux
    Ort::Env env;
    Ort::Session session_{env, "/home/wangzijian/Desktop/lite.ai.toolkit.win/models/lite/cv/CutOut/CutOut_fp32.onnx",
                          Ort::SessionOptions{nullptr}};

#elif _WIN32
    Ort::Env env;
    Ort::Session session_{nullptr};
Ort::Session session_{env, L"E:\\lite.ai.toolkit.win\\models\\lite\\cv\\CutOut\\CutOut_fp32.onnx",
                          Ort::SessionOptions{nullptr}};
#endif
};


/*
ModNet::ModNet() {

    std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
    //OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    ort_session = new Session(env, widestr.c_str(), sessionOptions);
    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++) {
        AllocatedStringPtr input_name_Ptr = ort_session->GetInputNameAllocated(i, allocator);
        input_names.push_back(input_name_Ptr.get());
    }
    for (int i = 0; i < numOutputNodes; i++) {
        AllocatedStringPtr output_name_Ptr = ort_session->GetInputNameAllocated(i, allocator);
        output_names.push_back(output_name_Ptr.get());
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }

}
*/