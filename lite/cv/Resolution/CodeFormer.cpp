//
// Created by wangzijian on 2023/9/7.
//

#include "CodeFormer.h"
using namespace Ort;


void CodeFormer::detect(std::string srcImg, std::string outImg) {
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

    // 记录输入维度
    int input_channels = input_dims.at(1);
    int input_height = input_dims.at(2);
    int input_width = input_dims.at(3);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_Ptr = session_.GetInputNameAllocated(0, allocator);
    const char *input_names[1] = {};
    input_names[0] = (input_name_Ptr.get());

    AllocatedStringPtr output_name_Ptr = session_.GetOutputNameAllocated(0, allocator);
    const char *output_names[1] = {};
    output_names[0] = (output_name_Ptr.get());

    cv::Mat test = cv::imread(srcImg);
    cv::Mat rgbTest;
    cv::cvtColor(test,rgbTest,cv::COLOR_BGR2RGB);
    cv::resize(rgbTest,rgbTest,cv::Size(input_width, input_height),0,0,cv::INTER_AREA);
    cv::Mat input;
    rgbTest.convertTo(input,CV_32FC3,1.f/255.f,-0.5f);
    input.convertTo(input,CV_32FC3,2.f,0);

    array<int64_t, 4> input_shape_{ 1, 3, input_height, input_width };
    vector<int64_t> input2_shape_ = { 1 };

    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<float> tensor_value_handler;
    std::vector<cv::Mat> mat_channels;
    cv::split(input,mat_channels);

    for (unsigned int i = 0; i < 3; ++i)
        std::memcpy(tensor_value_handler.data() + i * (input_height * input_width),
                    mat_channels.at(i).data, input_height * input_width * sizeof(float));

    auto input_tensor = Ort::Value::CreateTensor<float>(allocator_info,
                                                                (float *) tensor_value_handler.data(),
                                                                3 * input_height * input_width,
                                                                input_shape_.data(), input_shape_.size());



    vector<Value> ort_inputs;
    vector<double> input2_tensor;
    input2_tensor.push_back(0.5);
    ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, input_tensor.data(), input_tensor.size(), input_shape_.data(), input_shape_.size()));
    ort_inputs.push_back(Value::CreateTensor<double>(allocator_info, input2_tensor.data(), input2_tensor.size(), input2_shape_.data(), input2_shape_.size()));

}