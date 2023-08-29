//
// Created by wangzijian on 2023/8/22.
//

#include "LightEnhance.h"

// 进行前处理 我需要的只是vector向量而已
void LightEnhance::preprocessMat(cv::Mat inputMat, vector<float> &inputArray) {

    int h = inputMat.rows;
    int w = inputMat.cols;

    cv::cvtColor(inputMat, inputMat, cv::COLOR_BGR2RGB);

    cv::Mat floatTest(inputMat.size(), CV_32FC3);
    inputMat.convertTo(floatTest, CV_32FC3);
    cv::cvtColor(floatTest, floatTest, cv::COLOR_RGB2BGR);

    std::vector<float> floatVector;
    std::vector<cv::Mat> channels;
    cv::split(floatTest, channels);
    cv::Mat b, g, r;
    b = channels.at(0);
    g = channels.at(1);
    r = channels.at(2);

    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i]) / 255.0f;
    }
    cv::Mat input_test;
    cv::merge(channels, input_test);
    cv::cvtColor(input_test, input_test, cv::COLOR_BGR2RGB);

    int pos = 0;

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                // 访问Mat的方法
                float value = input_test.at<cv::Vec3f>(i, j)[c];
                inputArray[pos++] = value;
            }
        }
    }

}

void LightEnhance::runNet(std::vector<float> input, int h, int w, Ort::Value &outputTensor) {
    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};

    std::array<int64_t, 4> input_shape{1, 3, h, w};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUInput);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *) input.data(), 3 * h * w,
                                                        input_shape.data(), input_shape.size());

    session_.Run(Ort::RunOptions(), input_names, &input_tensor, 1, output_names, &outputTensor, 1);
}

void LightEnhance::postprocessMat(const float *outputDataPtr, int outW, int outH, cv::Mat &outPutMat) {

    std::vector<float> ort_output(outputDataPtr, outputDataPtr + outW * outH * 3);
    for (int i = 0; i < outW * outH * 3; ++i) {
        ort_output[i] = ort_output[i] * 255.0f;
        ort_output[i] = int(ort_output[i]);
        // 去掉错误的像素大小 如果有那种很离谱的颜色一般都是像素未截断
        if (ort_output[i] < 0) {
            ort_output[i] = 0;
        }
        if (ort_output[i] > 255) {
            ort_output[i] = 255;
        }
    }

    cv::Mat img(outH, outW, CV_8UC3);

    // 循环遍历像素
    for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {
            // BGR顺序写入
            img.at<cv::Vec3b>(y, x)[2] = ort_output[x + y * outW];
            img.at<cv::Vec3b>(y, x)[1] = ort_output[x + y * outW + outH * outW];
            img.at<cv::Vec3b>(y, x)[0] = ort_output[x + y * outW + outH * outW * 2];
        }
    }

    outPutMat = img;
}


void LightEnhance::detect(std::string srcImg, std::string outImg) {

    Mat test = cv::imread(srcImg);

    int h = test.rows;
    int w = test.cols;

    std::vector<float> inputArray(3 * h * w);

    preprocessMat(test, inputArray);

    runNet(inputArray, h, w, outputTensor);

    // 这行应该也可以封装进去
    const float *outputDataPtr = outputTensor.GetTensorData<float>();
    cv::Mat outputMat;

    postprocessMat(outputDataPtr, w, h, outputMat);

    cv::imwrite(outImg, outputMat);

}
