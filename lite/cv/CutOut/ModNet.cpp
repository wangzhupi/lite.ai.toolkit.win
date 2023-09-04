//
// Created by Administrator on 2023/8/22.
//

#include "ModNet.h"

void ModNet::get_scale_factor(std::vector<float> &scale, int im_h, int im_w, int ref_size) {

    int im_rw;
    int im_rh;
    if ((((im_h) > (im_w)) ? (im_h) : (im_w)) < ref_size || (((im_w) < (im_h)) ? (im_w) : (im_h)) > ref_size) {
        if (im_w >= im_h) {
            im_rh = ref_size;
            im_rw = int(float(im_w) / float(im_h) * float(ref_size));
        } else {
            im_rw = ref_size;
            im_rh = int(float(im_h) / float(im_w) * ref_size);
        }
    } else {
        im_rh = im_h;
        im_rw = im_w;
    }

    im_rw = im_rw - im_rw % 32;
    im_rh = im_rh - im_rh % 32;

    float x_scale_factor = im_rw / float(im_w);
    float y_scale_factor = im_rh / float(im_h);

    scale.push_back(x_scale_factor);
    scale.push_back(y_scale_factor);
}


// 进行前处理 我需要的只是vector向量而已
void ModNet::preprocessMat(cv::Mat img, vector<float> &inputArray, int &resize_h, int &resize_w) {

    // 得到浮点型的矩阵
    img.convertTo(img, CV_32FC3);
    vector<cv::Mat> channels;
    split(img, channels);

    Mat b, g, r;
    r = channels.at(0);
    g = channels.at(1);
    b = channels.at(2);

    // 进行前处理
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - 127.5f) / 127.5f;
    }

    cv::Mat normalizedImg;

    cv::merge(channels, normalizedImg);

    std::vector<float> get_scale;

    get_scale_factor(get_scale, normalizedImg.rows, normalizedImg.cols, 512);

    float x = get_scale[0];
    float y = get_scale[1];

    cv::resize(normalizedImg, normalizedImg, cv::Size(), x, y, cv::INTER_AREA);
    cv::cvtColor(normalizedImg, normalizedImg, cv::COLOR_BGR2RGB);

    int height_new = normalizedImg.rows;
    int width_new = normalizedImg.cols;

    std::vector<float> vec(3 * height_new * width_new);

    int pos = 0;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < height_new; i++) {
            for (int j = 0; j < width_new; j++) {
                // 访问Mat的方法
                float value = normalizedImg.at<cv::Vec3f>(i, j)[c];
                vec[pos++] = value;
            }
        }
    }
    resize_w = width_new;
    resize_h = height_new;
    inputArray = vec;
}

void ModNet::runNet(std::vector<float> input, int h, int w, Ort::Value &outputTensor) {
    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};

    std::array<int64_t, 4> input_shape{1, 3, h, w};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUInput);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *) input.data(), 3 * h * w,
                                                        input_shape.data(), input_shape.size());

    session_.Run(Ort::RunOptions(), input_names, &input_tensor, 1, output_names, &outputTensor, 1);

}


void ModNet::postprocessMat(const float *outputDataPtr, int resize_w, int resize_h, Mat &outputMat) {


    int totalNum = 1 * resize_h * resize_w;
    std::vector<float> output_tensor(totalNum);

    // 拷贝到output_tensor其中
    std::copy(outputDataPtr, outputDataPtr + totalNum, output_tensor.data());

    // 后处理
    for (int i = 0; i < totalNum; ++i) {
        output_tensor[i] = output_tensor[i] * 255.0f;
        output_tensor[i] = int(output_tensor[i]);
        output_tensor[i] = std::clamp(output_tensor[i], 0.f, 255.f);
    }

    cv::Mat outputImg(resize_h, resize_w, CV_8UC1);

    // 循环遍历像素
    for (int y = 0; y < resize_h; y++) {
        for (int x = 0; x < resize_w; x++) {
            // 只写入第一个通道
            outputImg.at<uchar>(y, x) = output_tensor[x + y * resize_w];
        }
    }

    // 原图大小
    cv::resize(outputImg, outputImg, cv::Size(outputMat.cols, outputMat.rows), 0, 0, cv::INTER_AREA);
    outputMat = outputImg;

    //    设置窗口名称
    //    cv::namedWindow("image", cv::WINDOW_NORMAL);
    //    cv::resizeWindow("image", 500, 500);
    //    cv::imshow("image", outputMat);
    //    cv::waitKey(0);


}

void ModNet::detect(string srcimg, string outImg) {
    // 读入图片
    Mat test = imread(srcimg);
    int h = test.rows;
    int w = test.cols;
    std::vector<float> inputArray;
    // 前处理
    int resize_h = 0;
    int resize_w = 0;
    preprocessMat(test, inputArray, resize_h, resize_w);

    // runNet
    Ort::Value outputTensor{nullptr};
    runNet(inputArray, resize_h, resize_w, outputTensor);

    // 获取输出shape的代码
    const float *outputDataPtr = outputTensor.GetTensorData<float>();
    std::vector<int64_t> shape = outputTensor.GetTensorTypeAndShapeInfo().GetShape();


    cv::Mat outputMat(h, w, CV_8UC3);


    postprocessMat(outputDataPtr, resize_w, resize_h, outputMat);

    cv::imwrite(outImg, outputMat);

}