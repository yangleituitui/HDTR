import os, cv2
import argparse
import face_alignment
import pickle, time
import shutil

from os.path import join
from tqdm import tqdm

import torch
import numpy as np 
from networks.models import Generator
from dataset.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
border = 2
#从视频中提取图像帧
split_image_cmd = 'ffmpeg -hwaccel cuvid -hide_banner -loglevel quiet -y -i {} -r 25 {}/%05d.png'
#从视频中提取音频
split_wav_cmd = 'ffmpeg -hwaccel cuvid -hide_banner -loglevel quiet -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'
#提取三维人脸关键点
fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda:0')

#加载预训练模型并返回该模型
def load_model(args):
    Model = Generator(args)
    print("Load checkpoint from: {}".format(args.model_path))
    checkpoint = torch.load(args.model_path)
    s = checkpoint#["state_dict"]   #加载模型参数
    new_s = {}              #创建一个新的字典 new_s，用于存储处理后的 state_dict
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v    #不是在多GPU上运行 可以去掉前缀

    Model.load_state_dict(new_s)     #处理后的 state_dict 加载到模型 Model 中
    Model = Model.cuda().eval()     #将模型转移到 CUDA 设备上并设置为 evaluation 模式。
    
    return Model


#在图像上绘制连接人脸关键点的线条
def pixel_connet(img, landmarks):
    # General key point connection
    for i in range(60, 67):
        cv2.line(img, landmarks[i].astype(int), landmarks[i+1].astype(int), 0, 1)
        
    # Conjunction successive key points, joint formation multiple variations
    cv2.line(img, landmarks[67].astype(int), landmarks[60].astype(int), 0, 1)
    return img

#根据面部关键点坐标来定位嘴巴区域 返回嘴部区域四个坐标值
def cutting_mouth(landmarks):
    landmarks = landmarks.astype(int)
    
    x_jaw, y_jaw = landmarks[8]    # Lower tomoe
    x_nose, y_nose = landmarks[30]   # nose tip
    x_left_mouth, y_left_mouth = landmarks[48]  # left beak angle
    x_right_mouth, y_right_mouth = landmarks[54]  # right beak angle

    if (y_jaw - y_nose) > (x_right_mouth - x_left_mouth):
        padlen = ((y_jaw - y_nose) - (x_right_mouth - x_left_mouth))/2
        x_left_mouth -= padlen
        x_right_mouth += padlen
    elif (y_jaw - y_nose) < (x_left_mouth - x_right_mouth):
        padlen = ((x_right_mouth - x_left_mouth) - (y_jaw - y_nose))/2
        y_nose -= padlen
        y_jaw += padlen
        
    return 	int(x_left_mouth), int(y_nose), int(x_right_mouth), int(y_jaw)

#生成面部口罩效果  唇部
def get_mask(image, landmarks):
    # cut_image = aligned_img[int(y_nose):int(y_jaw), int(x_left_mouth):int(x_right_mouth), :]
    # Difinate the mouth region landmarks indices
    mouth_indices = list(range(48, 60))  # mouth landmarks indices
    tooth_indices = list(range(60, 67))    # tooth landmarks indices

    # Create mouth region mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mouth_mask = np.zeros_like(gray)
    tooth_mask = np.zeros_like(gray)

    # Accroding to landmarks indices to build mouth region polygon
    mouth_ = np.array(landmarks[mouth_indices], dtype=np.int32)
    tooth_ = np.array(landmarks[tooth_indices], dtype=np.int32)
    cv2.fillPoly(mouth_mask, [mouth_], 255)
    cv2.fillPoly(tooth_mask, [tooth_], 255)
    
    # Set the mouth area to black
    mouth_area = cv2.bitwise_and(image, image, mask=mouth_mask)
    mouth_area[mouth_mask == 0] = 0

    tooth_area = cv2.bitwise_and(image, image, mask=tooth_mask)
    tooth_area[tooth_mask == 0] = 0

    lips_area = mouth_area - tooth_area
    lips_mask = mouth_mask - tooth_mask
    in_mask = image - mouth_area
    
    # mean_tooth = np.mean(image[np.where(tooth_mask == 255)])
    # mean_lip = np.mean(image[np.where(lips_mask == 255)])

    mean_tooth = np.mean(image[tooth_area != 0])
    mean_lip = np.mean(image[lips_area != 0])
    mean_mask = np.zeros_like(image)

    mean_mask[np.where(tooth_mask == 255)] = mean_tooth
    mean_mask[np.where(lips_mask == 255)] = mean_lip
    
    # mean_mask[tooth_mask != 0] = mean_tooth
    # mean_mask[lips_mask != 0] = mean_lip
    in_mean_mask = pixel_connet(mean_mask, landmarks)

    return in_mask, in_mean_mask, image
# def get_mask(image, landmarks):   #牙齿
#     # Difinate the mouth region landmarks indices
#     tooth_indices = list(range(60, 67))    # tooth landmarks indices

#     # Create tooth region mask
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     tooth_mask = np.zeros_like(gray)

#     # Accroding to landmarks indices to build tooth region polygon
#     tooth_ = np.array(landmarks[tooth_indices], dtype=np.int32)
#     cv2.fillPoly(tooth_mask, [tooth_], 255)
    
#     # Set the tooth area to black
#     tooth_area = cv2.bitwise_and(image, image, mask=tooth_mask)
#     tooth_area[tooth_mask == 0] = 0

#     mean_tooth = np.mean(image[tooth_area != 0])
#     mean_mask = np.zeros_like(image)
#     mean_mask[np.where(tooth_mask == 255)] = mean_tooth
    
#     in_mask = image - tooth_area
#     in_mean_mask = pixel_connet(mean_mask, landmarks)
#     # cv2.imwrite('/home/lcj/Project/yangl/HDTR-Net-main/mask_image/in_mask.jpg', in_mask)
#     # cv2.imwrite('/home/lcj/Project/yangl/HDTR-Net-main/mask_image/in_mean_mask.jpg', in_mean_mask)
#     # cv2.imwrite('/home/lcj/Project/yangl/HDTR-Net-main/mask_image/image.jpg', image)
#     return in_mask, in_mean_mask, image

#输入
#full_frame_path 原图像序列路径
#temp_dir 保存目录
#输出
#in_mask 遮罩图像, in_mean_mask 轮廓图像, cut_image 嘴部裁剪图像
#返回
#frame_list 图像序列列表
def prepare_data(full_frame_path, temp_dir):
    # Detect face landmarks 
    #检测关键点 并保存二维坐标
    print(f'==> detect face landmarks ...')
    frame_list = sorted(os.listdir(full_frame_path))

    landmarks_dict = {}
    for index in tqdm(frame_list):
        # index_base = index.replace('.jpg', '')
        image = cv2.imread(join(full_frame_path, index))
        try:
            preds = fa_3d.get_landmarks(image)  #获得图像关键点坐标 (68*3)
        except Exception as e:
            print(f'Catched the following error: {e}')
            preds = None
            continue
        lmark = preds[0][:,:2]       #提取(x,y)   (68*2) 
        landmarks_dict[index] = lmark  #创建帧索引 帧数：(x,y)
    
    
    landmarks_path = join(temp_dir, 'landmarks')  #保存 landmarks 文件的路径
    os.makedirs(landmarks_path, exist_ok=True)
    
    if os.path.exists(join(landmarks_path, 'landmarks.pkl')):
        os.remove(join(landmarks_path, 'landmarks.pkl'))

    with open(join(landmarks_path, 'landmarks.pkl'), 'wb') as f:  #
        pickle.dump(landmarks_dict, f)   #landmark的二进制文件

    # Crap based on landmarks 
    #align_crop_image_path，用来保存人脸对齐裁剪后的图片。
    # 然后它从之前保存的 landmarks.pkl 文件中读取关键点数据，并根据这些关键点数据进行人脸对齐裁剪的操作
    align_crop_image_path = join(temp_dir, 'align_crop_image')
    os.makedirs(align_crop_image_path, exist_ok=True)
    
    print(f'==> face align ...')
    with open(join(landmarks_path, 'landmarks.pkl'), 'rb') as f:
        read_landmarks = pickle.load(f)
    
    image_config = {}
    image_coor = {}
    image_hw = {}
    image_config_path = join(temp_dir, 'image_config')
    os.makedirs(image_config_path, exist_ok=True)
    if os.path.exists(join(image_config_path, 'image_config.pkl')):
        os.remove(join(image_config_path, 'image_config.pkl'))

    with open(join(image_config_path, 'image_config.pkl'), 'wb') as config:
        for i in tqdm(frame_list):   #遍历原图像序列
            # pdb.set_trace()
            landmark = read_landmarks[i]
            
            image_path = join(full_frame_path, i)
            image = cv2.imread(image_path)
            
            in_mask, in_mean_mask, cut_image = get_mask(image, landmark)
            
            x0, y0, x1, y1 = cutting_mouth(landmark)
            image_coor[i] = (x0, y0, x1, y1)
            
            #对图像的裁剪操作，主要是根据传入的坐标值对图像进行裁剪，并确保裁剪后的区域不超出图像的边界。
            h, w, _ = image.shape
            if x0 < 0: x0 = 0
            if y0 < 0: y0 = 0
            if x1 > w: x1 = w - 1
            if y1 > h: y1 = h - 1
            
            base_image = i.replace('.png', '')
            cv2.imwrite(join(align_crop_image_path, base_image + '_mask.png'), in_mask[y0:y1, x0:x1, :])
            cv2.imwrite(join(align_crop_image_path, base_image + '_mean_mask.png'), in_mean_mask[y0:y1, x0:x1, :])
            cv2.imwrite(join(align_crop_image_path, base_image + '_image.png'), cut_image[y0:y1, x0:x1, :])
            image_hw[i] = (y1 - y0, x1 - x0)
        image_config['image_hw'] = image_hw  #裁剪高度宽度
        image_config['image_coor'] = image_coor  #裁剪坐标
        pickle.dump(image_config, config)
    return frame_list

#将生成的超分辨率图像保存到指定路径下
#SR_images：表示要保存的超分辨率图像数据，是一个包含多张图像的数组。
#path：表示保存结果的根路径。
#index：表示起始索引值。
def save_results(SR_images, path, index):
    save_path = join(path, 'model_out')
    os.makedirs(save_path, exist_ok=True)
    
    for ind in range(SR_images.shape[0]):
        pred_hq = SR_images[ind, :].numpy() * 255.
        pred_hq = np.transpose(pred_hq.astype(int), (1, 2, 0))
    
        cv2.imwrite(join(save_path, '{}.png'.format(str(index+ind).zfill(5))), pred_hq)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default='/home/lcj/Project/yangl/HDTR-Net-main/sample/ZYnv_mouth_mouth_50.mp4', help='Input video to clear mouth region')
    parser.add_argument('--temp_dir', type=str, default='./test2', help='Temp directory to save output')
    parser.add_argument('--model_path', type=str, default='/home/lcj/Project/18T_another_one/yanglei/HDTR/checkpoint/zynv_mouth/gen/generator_weights_epoch50.pth', 
                        help='Root path of pretrained SR model')
    
    parser.add_argument("--mask_channel", type=int, default=6, help='Channel of the mouth mask and mouth contour')
    parser.add_argument("--ref_channel", type=int, default=3, help='Channel of the reference channel')
    parser.add_argument("--mouth_size", type=int, default=96, help='Size of the crop mouth region image')
    parser.add_argument("--test_batch_size", type=int, default=4, help='Size of the test batch')
    parser.add_argument("--test_workers", type=int, default=0, help='Number of workers to run the test')
    
    args = parser.parse_args()
    
    # --- Split image from video --- #
    video_basename = os.path.basename(args.input_video).replace('.mp4', '')  #视频基础名称 zhongyou 
    result_path = join(args.temp_dir, video_basename)   #结果路径   ./test_result/中邮
    
    if os.path.exists(result_path):  
        shutil.rmtree(result_path)    #清空
    
    os.makedirs(result_path, exist_ok=True)    
    full_frame_path = join(result_path, 'full_frame')    #原图像完整帧路径
    wav_path = join(result_path, 'audio.wav')   #音频路径
    os.makedirs(full_frame_path, exist_ok=True)
    #拆分图像的命令，将输入的视频文件分割成单独的帧图像，并保存在 full_frame_path 目录下
    os.system(split_image_cmd.format(args.input_video, full_frame_path))
    #执行了提取音频的命令，从输入的视频文件中提取音频并保存为 WAV 格式文件到 wav_path 路径下。
    os.system(split_wav_cmd.format(args.input_video, wav_path))

    # --- Prepare our input data --- #
    print(f'[info] Step1: Prepare input data ...')
    frame_list = prepare_data(full_frame_path, result_path) #生成mask cut 返回图像序列列表
    
    align_crop_image_path = join(result_path, 'align_crop_image')
    
    # --- Load model and inferece --- #
    print('[info] Step2: load model ...')
    HQ_Generator = load_model(args)
    
    #数据处理成pytorch张量
    dataset = InferenceDataset(frame_list, align_crop_image_path, args.mouth_size)
    #给定的数据集进行批量加载和处理
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            num_workers=int(args.test_workers),
                            drop_last=False)
    
    start = time.time()

    #print(f"[info] Step3: save results ...")
    start_index = 1
    num_baches=513
    # print("num**")
    # print(num_baches)
    for mouth_mask, mouth_contour, referece in dataloader:
        print(f'[info]: Processing cropped image to clear mouth region : {start_index}-{start_index+args.test_batch_size}')
        
        with torch.no_grad():
            print("shape***")
            print(mouth_mask.shape)
            print(mouth_contour.shape)
            print(referece.shape)
            pred_hq = HQ_Generator(mouth_mask, mouth_contour, referece).cpu()
            
        save_results(pred_hq, result_path, index=start_index)
        start_index += args.test_batch_size
 
    end = time.time()
    print(end - start)
    print("时间/每帧")
    print((end - start)/num_baches)
    
    save_path = join(result_path, 'model_out')
    #将指定的音频文件和图像序列合成为视频
    synthesis_CMD = "ffmpeg -y -loglevel warning " + \
            "-thread_queue_size 8192 -i {} " + \
            "-thread_queue_size 8192 -i {}/%05d.png " + \
            "-vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest {}/{}_{}.mp4"
    #将两个图像序列水平拼接为一个视频
    merged_CMD = "ffmpeg -hide_banner -y -loglevel warning " + \
            "-thread_queue_size 8192 -i {}/%05d.png " + \
            "-thread_queue_size 8192 -i {}/%05d.png " + \
            "-i {} " + \
            "-filter_complex hstack=inputs=2 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p {}/merge_{}.mp4"
    
    os.system(synthesis_CMD.format(wav_path, save_path, result_path, "out", video_basename))
    
    
    # --- InverAffineTransform --- # 
    #贴回原背景
    """
    info: fixed not rotated(no warpaffine) by default
    """
    print('[info] Step4: Inver Affine Transform (Being here means wearing paste back) ...')
    warpAffine_parm_path = join(result_path, 'warpAffine_param')
    
    image_config_path = join(result_path, 'image_config')
    with open(join(image_config_path, 'image_config.pkl'), 'rb') as f:
        image_config = pickle.load(f)
    print('==> paste background ...')
    for index in tqdm(frame_list):
        full_image = cv2.imread(join(full_frame_path, index))

        # paste bask
        h, w = image_config['image_hw'][index]
        
        out_image = cv2.imread(join(save_path, index))
        out_image = cv2.resize(out_image, (w, h))
        
        x0, y0, x1, y1 = image_config['image_coor'][index]
        
        # create mouth mask
        mask = np.zeros(full_image.shape[:2], dtype=np.uint8)
        souce_img = full_image.copy()
        souce_img[y0:y1, x0:x1, :] = out_image
        mask[y0:y1, x0:x1] = 255

        # seamless
        mixed_image = cv2.seamlessClone(souce_img, full_image, mask, (x0+w//2, y0+h//2), cv2.NORMAL_CLONE)

        # full_image[y0:y1, x0:x1, :] = out_image
        finout_image_path = join(result_path, 'fin_out')
        os.makedirs(finout_image_path, exist_ok=True)
        cv2.imwrite(join(finout_image_path, index), mixed_image)
        
    os.system(synthesis_CMD.format(wav_path, finout_image_path, result_path, "result", video_basename))
    os.system(merged_CMD.format(full_frame_path, finout_image_path, wav_path, result_path, video_basename))
    print(f"[info]: Result videos is saved in {result_path}")
    print("[info]: Animate over !!!")