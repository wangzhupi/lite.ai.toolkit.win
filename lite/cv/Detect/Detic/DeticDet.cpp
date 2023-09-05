//
// Created by wangzijian on 2023/9/5.
//

#include "DeticDet.h"

void DeticDet::preprocess(cv::Mat input) {
    int height = input.rows;
    int width = input.cols;
    int channels = input.channels();

    cv::Mat rgbInput;
    cv::cvtColor(input,rgbInput,cv::COLOR_BGR2RGB);

    float oh;
    float ow;

    if (height < width)
    {
        scale = float(max_size) / float(height);
        oh = float (max_size);
        ow = scale * float(width);
    }else{
        scale = float(max_size) / float(width);
        oh = scale * float(height);
        ow = max_size;
    }

}


void DeticDet::detect(std::string srcImg) {
    cv::Mat test = cv::imread(srcImg);
    int ori_w = test.cols;
    int ori_h = test.rows;
    preprocess(test);
}