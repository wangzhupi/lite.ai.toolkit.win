//
// Created by wangzijian on 2023/8/23.
//

#include "Photo2Cartoon.h"
#include "memory"

// 逻辑上没有问题 但是命名把我搞混了 后面将其封装起来
// 把那个nchw和nwhc改写封装为一个函数


// preProcessd
// photo->mask
void Photo2Cartoon::preprocessMat(cv::Mat inputMat, cv::Mat &normalized) {
    // 记录输入的长和宽 到时候resize要用
    int origin_h = inputMat.rows;
    int origin_w = inputMat.cols;

    cv::cvtColor(inputMat, inputMat, COLOR_BGR2RGB);
    cv::resize(inputMat, inputMat, cv::Size(384, 384), 0, 0, cv::INTER_AREA);
    inputMat.convertTo(inputMat, CV_32FC3);
    // 这里返回了归一化的Mat矩阵
    inputMat.convertTo(normalized, CV_32FC3, 1.f / 255.f, 0);
}

// photo -> cartoon
void Photo2Cartoon::preprocessMat(cv::Mat inputMat, cv::Mat mask, cv::Mat &mergedMat) {

    cv::Mat face;
    cv::resize(inputMat, face, cv::Size(256, 256), 0, 0, INTER_AREA);
    // face转为浮点数字
    face.convertTo(face, CV_32FC3, 1.f, 0.f);

    cv::resize(mask, mask, cv::Size(256, 256), 0, 0, INTER_AREA);

    // 将mask转为3通道的

    if (mask.channels() != 3) {
        cv::cvtColor(mask, mask, COLOR_GRAY2BGR);
    }

    mergedMat = face.mul(mask) + (1.0 - mask) * 255.f;

    mergedMat.convertTo(mergedMat, CV_32FC3, 1.f / 127.5f, -1.0f);

}

float *Photo2Cartoon::runNet(cv::Mat merged, cv::Mat mask) {

    std::array<int64_t, 4> input_shape_cartoon{1, 3, merged.rows, merged.cols};
    const unsigned int target_height_cartoon = input_shape_cartoon.at(2);
    const unsigned int target_width_cartoon = input_shape_cartoon.at(3);
    const unsigned int target_channel_cartoon = input_shape_cartoon.at(1);
    const unsigned int target_tensor_size_cartoon =
            target_channel_cartoon * target_height_cartoon * target_width_cartoon;
    std::vector<float> tensor_value_handler_cartoon(3 * merged.rows * merged.cols);
    tensor_value_handler_cartoon.resize(target_tensor_size_cartoon);


    std::vector<cv::Mat> mat_channels;
    cv::split(merged, mat_channels);
    for (unsigned int i = 0; i < 3; ++i)
        std::memcpy(tensor_value_handler_cartoon.data() + i * (target_height_cartoon * target_width_cartoon),
                    mat_channels.at(i).data, target_height_cartoon * target_width_cartoon * sizeof(float));
    auto memory_info_cartoon = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUInput);


    auto input_tensor_cartoon = Ort::Value::CreateTensor<float>(memory_info_cartoon,
                                                                (float *) tensor_value_handler_cartoon.data(),
                                                                3 * merged.rows * merged.rows,
                                                                input_shape_cartoon.data(), input_shape_cartoon.size());


    const char *input_names_cartoon[] = {"input"};
    const char *output_names_cartoon[] = {"output"};

    std::vector<Ort::Value> ort_output_cartoon = session_cartooin.Run(Ort::RunOptions(), input_names_cartoon,
                                                                      &input_tensor_cartoon, 1, output_names_cartoon,
                                                                      1);

    Ort::Value &cartoon_pred = ort_output_cartoon.at(0);
    auto cartoon_dims = cartoon_pred.GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int out_h_cartoon = cartoon_dims.at(2);
    const unsigned int out_w_cartoon = cartoon_dims.at(3);

    float *cartoon_ptr = cartoon_pred.GetTensorMutableData<float>();

    // TODO
    return cartoon_ptr;


}


float *Photo2Cartoon::runNet(cv::Mat normalizedMat) {

    std::vector<float> tensor_value_handler_seg(3 * normalizedMat.rows * normalizedMat.cols);

    // 如果是单输入单输出的话 我更推荐使用这种直接赋值名字的操作
    // 如果是多输入多输出 我推荐使用Modnet.h 中的循环取名字的方法 line 49
    const char *input_names[] = {"input_1:0"};
    const char *output_names[] = {"sigmoid/Sigmoid:0"};

    // 输入维度
    std::array<int64_t, 4> input_shape_seg{1, normalizedMat.rows, normalizedMat.cols, 3};

    const unsigned int target_height_seg = input_shape_seg.at(1);
    const unsigned int target_width_seg = input_shape_seg.at(2);
    const unsigned int target_channel_seg = input_shape_seg.at(3);
    const unsigned int target_tensor_size_seg = target_channel_seg * target_height_seg * target_width_seg;
    tensor_value_handler_seg.resize(target_tensor_size_seg);

    // 生成输入的vector
    std::memcpy(tensor_value_handler_seg.data(), normalizedMat.data, target_tensor_size_seg * sizeof(float));

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUInput);
    // 生成输入的Tensor
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *) tensor_value_handler_seg.data(),
                                                        3 * normalizedMat.rows * normalizedMat.cols,
                                                        input_shape_seg.data(), input_shape_seg.size());


    vector<Value> ort_outputs = session_.Run(Ort::RunOptions(), input_names, &input_tensor, 1, output_names, 1);


    Value &mask_pred = ort_outputs.at(0);

    auto mask_dims = mask_pred.GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int out_h = mask_dims.at(1);
    const unsigned int out_w = mask_dims.at(2);
    const unsigned int out_channnels = mask_dims.at(3);

    float *mask_ptr = mask_pred.GetTensorMutableData<float>();
    std::vector<int64_t> shape = mask_pred.GetTensorTypeAndShapeInfo().GetShape();

    return mask_ptr;

}

void Photo2Cartoon::postprocessMat_p2c(float *outputDataPtr, int out_w_cartoon, int out_h_cartoon, cv::Mat &outPutMat,
                                       cv::Mat mask) {

    vector<Mat> cartoon_channel_mats;
    Mat rmat(out_h_cartoon, out_w_cartoon, CV_32FC1, outputDataPtr);
    Mat gmat(out_h_cartoon, out_w_cartoon, CV_32FC1, outputDataPtr + out_h_cartoon * out_w_cartoon);
    Mat bmat(out_h_cartoon, out_w_cartoon, CV_32FC1, outputDataPtr + 2 * out_h_cartoon * out_w_cartoon);

    // 这个地方也可以先合并 再用convertTo来进行
    rmat = (rmat + 1) * 127.5;
    gmat = (gmat + 1) * 127.5;
    bmat = (bmat + 1) * 127.5;

    cartoon_channel_mats.push_back(rmat);
    cartoon_channel_mats.push_back(gmat);
    cartoon_channel_mats.push_back(bmat);

    Mat cartoon;
    merge(cartoon_channel_mats, cartoon);

    if (mask.channels() != 3)
    {
        cv::cvtColor(mask,mask,COLOR_GRAY2BGR);
    }

    cv::resize(mask, mask, cv::Size(256, 256), 0, 0, INTER_AREA);

    cartoon = cartoon.mul(mask) + (1.f - mask) * 255.f;
    cvtColor(cartoon, cartoon, COLOR_BGR2RGB);
    // 这个操作是为了什么
    cartoon.convertTo(outPutMat, CV_8UC3);

}


void Photo2Cartoon::detect1(std::string srcImg, std::string outImg) {

    // 前处理
    cv::Mat test = cv::imread(srcImg);
    //备份test
    cv::Mat testBak = test;
    int origin_h = test.rows;
    int origin_w = test.cols;
    cv::cvtColor(test, test, COLOR_BGR2RGB);
    cv::resize(test, test, cv::Size(384, 384), 0, 0, cv::INTER_AREA);
    cv::Mat floatTest;
    test.convertTo(floatTest, CV_32FC3);
    std::vector<cv::Mat> channels;
    cv::Mat b, g, r;
    cv::split(floatTest, channels);



//
//    // 这个时候已经进行了bgr2rgb的转换
//    for (int i = 0; i < 3; i++) {
//        channels[i] = channels[i] / 255.0f;
//    }
//    cv::Mat normalizedMat;
//    cv::merge(channels, normalizedMat);

    cv::Mat normalizedMat;
    floatTest.convertTo(normalizedMat, CV_32FC3, 1.f / 255.f, 0);




    // run Net
    std::vector<float> tensor_value_handler(3 * floatTest.rows * floatTest.cols);

    // get input name
    // 这一段其实可以封装到头文件中
    AllocatorWithDefaultOptions allocator;
    AllocatedStringPtr input_name_Ptr = session_.GetInputNameAllocated(0, allocator);
    const char *input_names[1] = {};
    input_names[0] = (input_name_Ptr.get());

    // get output name
    AllocatedStringPtr output_name_Ptr = session_.GetOutputNameAllocated(0, allocator);
    const char *output_names[1] = {};
    output_names[0] = (output_name_Ptr.get());


    // 输入维度
    std::array<int64_t, 4> input_shape{1, floatTest.rows, floatTest.cols, 3};

    // 处理NWHC的方法
    // https://zhuanlan.zhihu.com/p/524230808
    const unsigned int target_height = input_shape.at(1);
    const unsigned int target_width = input_shape.at(2);
    const unsigned int target_channel = input_shape.at(3);
    const unsigned int target_tensor_size = target_channel * target_height * target_width;
    tensor_value_handler.resize(target_tensor_size);

    std::memcpy(tensor_value_handler.data(), normalizedMat.data, target_tensor_size * sizeof(float));


    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUInput);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *) tensor_value_handler.data(),
                                                        3 * floatTest.rows * floatTest.cols,
                                                        input_shape.data(), input_shape.size());

//    session_.Run(Ort::RunOptions(), input_names, &input_tensor, 1, output_names, &outputTensor, 1);


    vector<Value> ort_outputs = session_.Run(Ort::RunOptions(), input_names, &input_tensor, 1, output_names, 1);


    // 这里还是得有待商榷一下
    // 这里的为什么输出的通道数为1 原因在于分析其onnx的模型可以看到他的输出的时候通道数就是1
    // 所以其他的都对的上了 输出维度主要是由onnx来决定
    // 我们只需要对准输入维度和对于输出的后处理
    Value &mask_pred = ort_outputs.at(0);

    auto mask_dims = mask_pred.GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int out_h = mask_dims.at(1);
    const unsigned int out_w = mask_dims.at(2);
    const unsigned int out_channnels = mask_dims.at(3);

    float *mask_ptr = mask_pred.GetTensorMutableData<float>();

    // 更为优雅的写法
    // 他这里直接利用初始化Mat进行构造
    // 但是这里有个问题就是他这里的GetTensorMutabledata的作用如何使用
    // 这里的应该是直接拷贝的 没有使用for循环来赋值这样更快更高效

    // postprocess
    // 这样直接拷贝指针的数据到Mat的效率会更快
    Mat mask;
    Mat mask_out(out_h, out_w, CV_32FC1, mask_ptr);
    resize(mask_out, mask, Size(origin_w, origin_h));
    // 这样的方法更简单不用再写循环
    // 使得每个元素都乘以255之后取整
    // 多通道的方法应该和这个类似

    // convertTo的后两个参数的含义为缩放和偏移的操作
    // 缩放的含义就是相乘相除
    // 你设置为255就是乘以255
    // 设置为1/255 就是除以255
    cv::Mat outputMask;
    mask.convertTo(outputMask, CV_8UC1, 255, 0);
    cv::resize(outputMask, outputMask, cv::Size(256, 256));
    int channels1 = outputMask.channels();
    // 如果不是的话那么将其转化为3通道的
    if (channels1 != 3) {
        cv::cvtColor(outputMask, outputMask, COLOR_GRAY2BGR);
    }
    // 又进行归一化 这里的可以之后用这种方法来进行归一化
    // 尽量少用循环的方法
    // 这里有重复操作 删除掉不需要的
    cv::Mat normalized;
    outputMask.convertTo(normalized, CV_32F, 1.0 / 255.0);



    // 处理photo2cartoon
    cv::Mat face;
    cv::resize(testBak, face, cv::Size(256, 256), 0, 0, cv::INTER_AREA);
    face.convertTo(face, CV_32FC3, 1.0f, 0.0f);

    cv::Mat mat_merged_rs = face.mul(normalized) + (1.f - normalized) * 255.f;
    // convertTo的第一个参数是乘除
    // 第二个是加减
    mat_merged_rs.convertTo(mat_merged_rs, CV_32FC3, 1.f / 127.5f, -1.f);

    // 预处理完成现在进行cartoon的推理阶段

    // 做inputTensor 这次的输入格式为NCHW
    // 这种写函数封装起来
    std::array<int64_t, 4> input_shape_cartoon{1, 3, mat_merged_rs.rows, mat_merged_rs.cols};
    const unsigned int target_height_cartoon = input_shape_cartoon.at(2);
    const unsigned int target_width_cartoon = input_shape_cartoon.at(3);
    const unsigned int target_channel_cartoon = input_shape_cartoon.at(1);
    const unsigned int target_tensor_size_cartoon = target_channel * target_height * target_width;
    std::vector<float> tensor_value_handler_cartoon(3 * mat_merged_rs.rows * mat_merged_rs.cols);
    tensor_value_handler_cartoon.resize(target_tensor_size_cartoon);


    std::vector<cv::Mat> mat_channels;
    cv::split(mat_merged_rs, mat_channels);
    for (unsigned int i = 0; i < 3; ++i)
        std::memcpy(tensor_value_handler_cartoon.data() + i * (target_height_cartoon * target_width_cartoon),
                    mat_channels.at(i).data, target_height_cartoon * target_width_cartoon * sizeof(float));
    auto memory_info_cartoon = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUInput);

    auto input_tensor_cartoon = Ort::Value::CreateTensor<float>(memory_info,
                                                                (float *) tensor_value_handler_cartoon.data(),
                                                                3 * mat_merged_rs.rows * mat_merged_rs.rows,
                                                                input_shape_cartoon.data(), input_shape_cartoon.size());


    const char *input_names_cartoon[] = {"input"};
    const char *output_names_cartoon[] = {"output"};

    std::vector<Ort::Value> ort_output_cartoon = session_cartooin.Run(Ort::RunOptions(), input_names_cartoon,
                                                                      &input_tensor_cartoon, 1, output_names_cartoon,
                                                                      1);

    Ort::Value &cartoon_pred = ort_output_cartoon.at(0);
    auto cartoon_dims = cartoon_pred.GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int out_h_cartoon = cartoon_dims.at(2);
    const unsigned int out_w_cartoon = cartoon_dims.at(3);

    float *cartoon_ptr = cartoon_pred.GetTensorMutableData<float>();

    // postprocess
    vector<Mat> cartoon_channel_mats;
    Mat rmat(out_h_cartoon, out_w_cartoon, CV_32FC1, cartoon_ptr);
    Mat gmat(out_h_cartoon, out_w_cartoon, CV_32FC1, cartoon_ptr + out_h_cartoon * out_w_cartoon);
    Mat bmat(out_h_cartoon, out_w_cartoon, CV_32FC1, cartoon_ptr + 2 * out_h_cartoon * out_w_cartoon);

    // 这个地方也可以先合并 再用convertTo来进行
    rmat = (rmat + 1) * 127.5;
    gmat = (gmat + 1) * 127.5;
    bmat = (bmat + 1) * 127.5;

    cartoon_channel_mats.push_back(rmat);
    cartoon_channel_mats.push_back(gmat);
    cartoon_channel_mats.push_back(bmat);

    Mat cartoon;
    merge(cartoon_channel_mats, cartoon);


    cartoon = cartoon.mul(normalized) + (1.f - outputMask) * 255.f;
    cvtColor(cartoon, cartoon, COLOR_BGR2RGB);
    // 这个操作是为了什么
    cartoon.convertTo(cartoon, CV_8UC3);


}


void Photo2Cartoon::detect(std::string srcImg, std::string outImg) {
    cv::Mat test = cv::imread(srcImg);

    // 记录原始传入的长和宽
    int ori_w = test.cols;
    int ori_h = test.rows;

    cv::Mat normalized;

    preprocessMat(test, normalized);

    // 利用返回的指针来进行处理任务
    auto mask_ptr = runNet(normalized);
    // 得到mask的缩略图
    cv::Mat mask(384, 384, CV_32FC1, mask_ptr);
    // 释放指针
    cv::Mat mask_out;
    cv::resize(mask, mask_out, cv::Size(ori_w, ori_h), 0, 0, INTER_AREA);

    // 进行photo2cartoon的预处理
    cv::Mat merged;
    preprocessMat(test, mask, merged);

    float *cartoon_ptr = runNet(merged, mask);
    Mat cartoonMat;
    postprocessMat_p2c(cartoon_ptr,256,256,cartoonMat,mask);

    cv::imwrite(outImg,cartoonMat);
}