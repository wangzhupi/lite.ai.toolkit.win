//
// Created by Administrator on 2023/8/22.
//

#include "utils.h"
#include "iostream"


using namespace std;

namespace Utils {
    void Utils::utils::runNet(std::vector<float> input, int h, int w, Ort::Value &outputTensor) {
        const char *input_names[] = {"input"};
        const char *output_names[] = {"output"};

        std::array<int64_t, 4> input_shape{1, 3, h, w};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPUInput);
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float *) input.data(), 3 * h * w,
                                                            input_shape.data(), input_shape.size());

        Ort::Session session_{nullptr};
        session_.Run(Ort::RunOptions(), input_names, &input_tensor, 1, output_names, &outputTensor, 1);

    }
}
