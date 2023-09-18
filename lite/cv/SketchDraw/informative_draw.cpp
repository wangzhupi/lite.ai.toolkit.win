//
// Created by wangzijian on 2023/9/7.
//

#include "informative_draw.h"
#include "string"
#include "vector"

using namespace Ort;


float *Informative_Draw::runNet(cv::Mat test, std::vector<int> &shape) {
    // 获取当前onnx的输入维度
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

    // 记录输入维度
    int input_channels = input_dims.at(1);
    int input_height = input_dims.at(2);
    int input_width = input_dims.at(3);

    shape.push_back(input_channels);
    shape.push_back(input_height);
    shape.push_back(input_width);

    cv::resize(test, test, cv::Size(input_width, input_height), 0, 0, cv::INTER_AREA);
    test.convertTo(test, CV_32FC3, 1.f, 0.f);

    // 属于nchw输入
    std::array<int64_t, 4> input_shape{1, input_channels, input_height, input_width};

    std::vector<float> tensor_value_handler(3 * input_height * input_width);

    std::vector<cv::Mat> mat_channels;
    cv::split(test, mat_channels);

    for (unsigned int i = 0; i < 3; ++i)
        std::memcpy(tensor_value_handler.data() + i * (input_height * input_width),
                    mat_channels.at(i).data, input_height * input_width * sizeof(float));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUInput);

    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *) tensor_value_handler.data(),
                                                        3 * input_height * input_width,
                                                        input_shape.data(), input_shape.size());


    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_Ptr = session_.GetInputNameAllocated(0, allocator);
    const char *input_names[1] = {};
    input_names[0] = (input_name_Ptr.get());

    AllocatedStringPtr output_name_Ptr = session_.GetOutputNameAllocated(0, allocator);
    const char *output_names[1] = {};
    output_names[0] = (output_name_Ptr.get());


    std::vector<Ort::Value> ort_output = session_.Run(Ort::RunOptions(), input_names,
                                                      &input_tensor, 1, output_names,
                                                      1);

    Ort::Value &pred = ort_output.at(0);

    float *output_ptr = pred.GetTensorMutableData<float>();

    return output_ptr;
}


void Informative_Draw::postprocess(float *output_ptr, std::vector<int> shape,std::string outImg,int ori_h,int ori_w) {

    std::vector<cv::Mat> channel_mats;
    int input_channels = shape.at(0);
    int input_height = shape.at(1);
    int input_width = shape.at(2);


    cv::Mat rmat(input_height, input_width, CV_32FC1, output_ptr);
    cv::Mat gmat(input_height, input_width, CV_32FC1, output_ptr + input_height + input_width);
    cv::Mat bmat(input_height, input_width, CV_32FC1, output_ptr + 2 * input_height + input_width);

    channel_mats.push_back(rmat);
    channel_mats.push_back(gmat);
    channel_mats.push_back(bmat);

    cv::Mat output;
    cv::merge(channel_mats, output);
    output.convertTo(output, CV_8UC3, 255, 0);
    cv::resize(output,output,cv::Size(ori_w,ori_h));

    cv::imwrite(outImg,output);

}


void Informative_Draw::detect(std::string srcImg, std::string outImg) {
    cv::Mat test = cv::imread(srcImg);
    int ori_height = test.rows;
    int ori_width = test.cols;
    float *output_ptr = runNet(test, shape);
    postprocess(output_ptr,shape,outImg,ori_height,ori_width);

}
