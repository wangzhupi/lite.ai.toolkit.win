import shutil
import time
import os
import numpy as np
import onnxruntime
import cv2
from tqdm import tqdm
import argparse
from math import ceil

def chop_forward_dim4(t_im1, ort_session, shave=8, min_size=160000):
    b, c, h, w = t_im1.shape
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    mod_size = 4
    if h_size % mod_size:
        h_size = ceil(h_size/mod_size)*mod_size  # The ceil() function returns the uploaded integer of a number
    if w_size % mod_size:
        w_size = ceil(w_size/mod_size)*mod_size
    inputlist = [
        t_im1[:, :, 0:h_size, 0:w_size],
        t_im1[:, :, 0:h_size, (w - w_size):w],
        t_im1[:, :, (h - h_size):h, 0:w_size],
        t_im1[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(4):
            input_batch = inputlist[i] 
            # output_batch = model(input_batch)
            ort_inputs = {'input': input_batch}
            ort_output = ort_session.run(None, ort_inputs)[0]
            outputlist.append(ort_output)  # In this mechine, we only have one GPU
    else:
        outputlist = [
            chop_forward_dim4(patch, ort_session, shave, min_size) \
                for patch in inputlist]
    scale = 1
    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output_ht = np.zeros([b, c, h, w])
    # 拼接
    output_ht[:, :, 0:h_half, 0:w_half] = outputlist[0][:, :, 0:h_half, 0:w_half]
    output_ht[:, :, 0:h_half, w_half:w] = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output_ht[:, :, h_half:h, 0:w_half] = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output_ht[:, :, h_half:h, w_half:w] = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    return output_ht

def process(args):
    flag_gpu = args.gpu
    if flag_gpu:
        dml_option = {'device_id': 0}
        device_dml = [('DmlExecutionProvider', dml_option), 'CPUExecutionProvider']
        sess_options = onnxruntime.SessionOptions()
        sess_options.enable_mem_pattern = False
        ort_session = onnxruntime.InferenceSession("lle_denoise_dynamic.onnx", sess_options, providers=device_dml)
    else:
        device_cpu = ['CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession("lle_denoise_dynamic.onnx", providers=device_cpu)
    img_path = 'input3'
    result_path ='result3'
    os.makedirs(result_path, exist_ok=True)

    for path in tqdm(sorted(os.listdir(img_path))):
        img = cv2.imread(os.path.join(img_path,path), cv2.IMREAD_UNCHANGED)
        img_test = np.transpose(img)
        img_test = np.swapaxes(img_test,1,2)
        img_test = np.expand_dims(img_test,0)

        #
        # img2 = cv2.imread("D:\\Users\Desktop\inference_model\\result1\\test_output.jpg")
        # img_test2 = np.transpose(img2)
        # img_test2 = np.swapaxes(img_test2,1,2)
        # img_test2 = np.expand_dims(img_test2,0)

        t0 = time.time()
        img = (img/255.0).astype(np.float32)

        input_img = np.transpose(img, [2, 0, 1])


        input_img = np.expand_dims(input_img, 0)
        # ort_inputs = {'input': input_img}
        # ort_output = ort_session.run(None, ort_inputs)[0]
        ort_output = chop_forward_dim4(input_img, ort_session)
        ort_output = np.squeeze(ort_output, 0)

        # with open('D:\\Users\\Desktop\\inference_model\\output07121111111111.txt', 'w') as file:
        #     # 遍历每个通道
        #     for channel in range(ort_output.shape[0]):
        #         # 遍历每行
        #         for row in range(ort_output.shape[1]):
        #             # 遍历每列
        #             for col in range(ort_output.shape[2]):
        #                 # 获取元素值
        #                 element = "%.7f" % ort_output[channel, row, col]
        #                 # 写入文件
        #                 file.write(str(element))
        #                 # 换行
        #                 file.write('\n')


        ort_output = np.clip(ort_output*255, 0, 255).astype(np.uint8)

        # with open('D:\\Users\\Desktop\\inference_model\\python_opt_final0718.txt', 'w') as file:
        #     # 遍历每个通道
        #     for channel in range(ort_output.shape[0]):
        #         # 遍历每行
        #         for row in range(ort_output.shape[1]):
        #             # 遍历每列
        #             for col in range(ort_output.shape[2]):
        #                 # 获取元素值
        #                 element = int(ort_output[channel, row, col])
        #                 # 写入文件
        #                 file.write(str(element))
        #                 # 换行
        #                 file.write('\n')

        ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)

        # img2 = cv2.imread("D:\\Users\\Desktop\\inference_model\\output_test.png");

        cv2.imwrite(os.path.join(result_path,path.split('.')[0]+'.png'),ort_output)
        # print('spending time is:',time.time()-t0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='use cpu', default=True)
    args = parser.parse_args()

    process(args)
