//
// Created by wangzijian on 2023/8/22.
//

#include "opencv2/opencv.hpp"
#include "onnxruntime_cxx_api.h"
#include "iostream"
#include "utils/utils.h"
#include "lite/cv/CutOut/ModNet.h"
#include "lite/cv/LightEnhance/LightEnhance.h"
#include "lite/cv/Photo2Cartoon/Photo2Cartoon.h"
#include "lite/cv/Detect/Detic/DeticDet.h"
#include "lite/cv/SketchDraw/informative_draw.h"
#include "lite/cv/Resolution/CodeFormer.h"



using namespace std;


int main() {


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

    // 人像分割测试
    ModNet modNet;
    string srcImg = "/home/wangzijian/Desktop/lite.ai.toolkit.win/resource/modnetTest.jpg";
    string outImg = "/home/wangzijian/Desktop/lite.ai.toolkit.win/result/modnetTest_linux3.jpg";
    modNet.detect(srcImg, outImg);

    // 暗光增强测试
    LightEnhance lightEnhance;
    string srcImg1 = "/home/wangzijian/Desktop/lite.ai.toolkit.win/resource/lightenhanceTest.jpg";
    string outImg1 = "/home/wangzijian/Desktop/lite.ai.toolkit.win/result/lightenhanceTest_linux3.jpg";
    lightEnhance.detect(srcImg1, outImg1);

    // photocartoon
    Photo2Cartoon photo2Cartoon;
    string srcImg2 = "/home/wangzijian/Desktop/lite.ai.toolkit.win/resource/photo2cartoonTest.jpg";
    string outImg2 = "/home/wangzijian/Desktop/lite.ai.toolkit.win/result/photo2cartoonTest_linux3.jpg";
//    photo2Cartoon.detect1(srcImg2, outImg2);

    photo2Cartoon.detect(srcImg2, outImg2);

    // 测试通过
    // 待完善
    DeticDet deticDet;
    string srcImg3 = "/home/wangzijian/Desktop/lite.ai.toolkit.win/resource/desk.jpg";
    deticDet.detect(srcImg3);

    // test

    // 这里有三个模型可以选择
    // 素笔画
    Informative_Draw informativeDraw;
    string srcImg4 = "/home/wangzijian/Desktop/lite.ai.toolkit.win/resource/drawTest.png";
    string outImg4 = "/home/wangzijian/Desktop/lite.ai.toolkit.win/result/DrawingsTest_linux3.jpg";
    informativeDraw.detect(srcImg4, outImg4);

    // 超分模型
    CodeFormer codeFormer;
    string srcImg5 = "/home/wangzijian/Desktop/lite.ai.toolkit.win/resource/codefomerTest.png";
    string outImg5 = "/home/wangzijian/Desktop/lite.ai.toolkit.win/result/codefomerTest_linux.jpg";
    codeFormer.detect(srcImg5,outImg5);


    cout << "hello lite ai" << endl;

#endif


}