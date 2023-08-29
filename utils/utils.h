//
// Created by wangzijian on 2023/8/22.


//  Todo 用于实现一些基本的功能
//  1. 前处理
//  2. 后处理
//  3. 将Net封装到一个函数中

#include "vector"
#include "string"
#include "onnxruntime_cxx_api.h"

namespace Utils {
    class utils {
    public:
        std::string modelPath;

        // 前处理
        virtual void preProcess();

        // 后处理
        virtual void postProcess();

        // 封装运行网络
        void runNet(std::vector<float> input, int h, int w, Ort::Value &outputTensor);
    private:
        Ort::Env env;
        Ort::Session session_{env, L"E:\\lite.ai.toolkit.win\\models\\lite\\cv\\CutOut\\CutOut_fp32.onnx",
                                       Ort::SessionOptions{nullptr}};
    };

}


