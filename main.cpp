//
// Created by Administrator on 2023/8/22.
//

#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "iostream"
#include "utils/utils.h"
#include "lite/cv/CutOut/ModNet.h"
#include "lite/cv/LightEnhance/LightEnhance.h"

using namespace std;


int main() {

    // git测试

    // 测试Modent

#ifdef _WIN32
    ModNet modNet;
    string srcImg = "D:\\Users\\Desktop\\modnet_tiny_onnx\\test.jpg";
    string outImg = "E:\\lite.ai.toolkit.win\\result\\test_modnet.jpg";
    modNet.detect(srcImg,outImg);

    LightEnhance lightEnhance;
    string srcImg1 = "D:\\Users\\Desktop\\inference_model\\input\\1.jpg";
    string outImg1 = "E:\\lite.ai.toolkit.win\\result\\test_lightenhance.jpg";
    lightEnhance.detect(srcImg1, outImg1);
    cout << "hello lite ai" << endl;
#elif linux

    // 图片修改
    ModNet modNet;
    string srcImg = "";
    string outImg = "";
    modNet.detect(srcImg,outImg);

    LightEnhance lightEnhance;
    string srcImg1 = "";
    string outImg1 = "";
    lightEnhance.detect(srcImg1, outImg1);
    cout << "hello lite ai" << endl;

#endif


}