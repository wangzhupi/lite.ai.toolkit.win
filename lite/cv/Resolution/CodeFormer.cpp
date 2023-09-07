//
// Created by wangzijian on 2023/9/7.
//

#include "CodeFormer.h"
using namespace Ort;

// 构造函数初始化
CodeFormer::CodeFormer() {

//    size_t numInputNodes = session_.GetInputCount();
//    size_t numOutputNodes = session_.GetOutputCount();
//    AllocatorWithDefaultOptions allocator;
//
//    for (int i = 0; i < numInputNodes; i++)
//    {
//        AllocatedStringPtr input_name_Ptr = session_.GetInputNameAllocated(i, allocator);
//        input_names.push_back(input_name_Ptr.get());
//        Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
//        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
//        auto input_dims = input_tensor_info.GetShape();
//        input_node_dims.push_back(input_dims);
//    }
//
//    for (int i = 0; i < numOutputNodes; i++)
//    {
//        AllocatedStringPtr output_name_Ptr = session_.GetOutputNameAllocated(i,allocator);
//        output_names.push_back(output_name_Ptr.get());
//        Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(i);
//        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
//        auto output_dims = output_tensor_info.GetShape();
//        output_node_dims.push_back(output_dims);
//    }


    // TODO掌握多输入的案例

    const char* input1 = "input.1";
    const char* input2 = "onnx::Cast_1";

    const char* output1 = "2936";
    const char* output2 = "logits";
    const char* output3 = "style_feat";

    input_names.push_back(const_cast<char*>(input1));
    input_names.push_back(const_cast<char*>(input2));

    output_names.push_back(const_cast<char*>(output1));
    output_names.push_back(const_cast<char*>(output2));
    output_names.push_back(const_cast<char*>(output3));

}



void CodeFormer::detect(std::string srcImg, std::string outImg) {
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto input_dims = input_type_info.GetTensorTypeAndShapeInfo().GetShape();

    // 记录输入维度
    int input_channels = input_dims.at(1);
    int input_height = input_dims.at(2);
    int input_width = input_dims.at(3);


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
    tensor_value_handler.resize(3 * input_height *input_width);
    std::vector<cv::Mat> mat_channels;
    cv::split(input,mat_channels);

    for (unsigned int i = 0; i < 3; ++i)
        std::memcpy(tensor_value_handler.data() + i * (input_height * input_width),
                    mat_channels.at(i).data, input_height * input_width * sizeof(float));

    //
    //    auto input_tensor = Ort::Value::CreateTensor<float>(allocator_info,
    //                                                                (float *) tensor_value_handler.data(),
    //                                                                3 * input_height * input_width,
    //                                                                input_shape_.data(), input_shape_.size());



    vector<Value> ort_inputs;
    vector<double> input2_tensor;
    input2_tensor.push_back(0.5);
    // CreateTensor的时候是用的vector
    ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, tensor_value_handler.data(), tensor_value_handler.size(), input_shape_.data(), input_shape_.size()));
    ort_inputs.push_back(Value::CreateTensor<double>(allocator_info, input2_tensor.data(), input2_tensor.size(), input2_shape_.data(), input2_shape_.size()));


    vector<Value> ort_outputs = session_.Run(RunOptions{ nullptr }, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names.data(), output_names.size());

    float* pred = ort_outputs[0].GetTensorMutableData<float>();

    const int channel_step = input_height * input_width;
    std::vector<cv::Mat> channelsMat;

    cv::Mat rmat(input_height, input_width, CV_32FC1, pred);
    cv::Mat gmat(input_height, input_width, CV_32FC1, pred + channel_step);
    cv::Mat bmat(input_height, input_width, CV_32FC1, pred + 2 * channel_step);

    channelsMat.push_back(rmat);
    channelsMat.push_back(gmat);
    channelsMat.push_back(bmat);

    cv::Mat output;
    cv::merge(channelsMat,output);


    // 这行代码的作用是将 mask 图像中小于阈值 -1 的像素值都设置为 -1，从而将图像中的特定像素区域限制在 -1 到 1 的范围内
    output.setTo(min_max[0],output<min_max[0]);
    output.setTo(min_max[1],output>min_max[1]);

    output = (output - min_max[0]) / (min_max[1] - min_max[0]);

    output *= 255.0f;

    output.convertTo(output,CV_8UC3,1,0);

    cv::cvtColor(output,output,cv::COLOR_BGR2RGB);

    // TODO
    // 为什么这句和上面那句不一样
    // 这个要明天核对一下

    output.convertTo(output,CV_32FC3,1.0f,min_max[0]);

    output.convertTo(output,CV_32FC3,1.f/2.f,0);

    output.convertTo(output,CV_32FC3,255.f,0);

    output.convertTo(output,CV_8UC3,1,0);

    cv::cvtColor(output,output,cv::COLOR_BGR2RGB);




}