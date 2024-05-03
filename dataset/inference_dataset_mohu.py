import cv2
import torch
from os.path import join
from tqdm import tqdm
import os
import random
import numpy as np
from glob import glob
    
#构建推理时的数据集
class InferenceDataset():
    #初始化数据集对象
    #frame_list 图像文件列表
    #crop_image_path 裁剪后的图像存储路径
    #mouth_size 嘴部区域的大小
    def __init__(self,folder_path,mouth_size):
        
        self.folder_path = folder_path
        self.mouth_size = mouth_size
        self.listlen=0
        self.mouth_masks, self.mouth_contours, self.refereces,self.cans= self.make_dataset(folder_path)

    #生成数据集，遍历所有图像文件，读取对应的裁剪后的嘴部蒙版、轮廓和参考图像，并返回这些数据
    def make_dataset(self, folder_path):  #只调用一次在init
        
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        mouth_masks = []
        mouth_contours = []
        refereces = []
        cans=[]
        for subfolder in subfolders:   #data_result/train/00001
            mouth_mask = []
            mouth_contour = []
            referece = []
            can=[]
            
            full_frame_path = os.path.join(subfolder, 'full_frame')   
            print(full_frame_path)                        #data_result/train/00001/full_frame
            full_frame_list=sorted(os.listdir(full_frame_path))
            crop_image_path = join(subfolder, 'align_crop_image')   
            if not os.path.exists(full_frame_path) or not os.path.exists(crop_image_path):
                print("not exists continue!!!!")
                continue
            #print("crop_image_path****")          #data_result/train/00001/align_crop_image
            #self.listlen=self.listlen+len(full_frame_list) 
            # print("crop_image_path****")
            # print(crop_image_path)
            mohu_path="/home/lcj/Project/18T_another_one/yanglei/zy_nan/mohu_mask/zynan_mohu/align_crop_image"
            #can_paths=glob(f"{crop_image_path}/*_image.png")
            can_paths=glob(f"{mohu_path}/*_image.png")
            # print("can_paths*****")
            # print(can_paths)
            for frame in full_frame_list:
                #print("frame****")
                #print(frame)   #00001.png
                basename = frame.replace('.png', '')
        
                mask_frame_path = join(crop_image_path, basename + '_mask.png')
                # print("**")
                # print(mask_frame_path)
                mean_mask_frame_path = join(crop_image_path, basename + '_mean_mask.png')
                # fixed or not reference image
                #ref_frame_path = random.choice(glob(join(align_image_path, basename + '_image.png')))
                ref_frame_path = join(crop_image_path, basename + '_image.png')
                
                if not os.path.exists(mask_frame_path) or not os.path.exists(mean_mask_frame_path)\
                            or not os.path.exists(ref_frame_path):
                    print("not exists break!!!!")
                    break
                
                can_frame_path=random.choice(can_paths)
                #print("can_frame_path****")
                #print(can_frame_path)
                
                
                mask_frame = cv2.imread(mask_frame_path)
                mean_mask_frame = cv2.imread(mean_mask_frame_path)
                ref_frame = cv2.imread(ref_frame_path)
                can_frame=cv2.imread(can_frame_path)
                self.listlen=self.listlen+1
                
                mouth_mask.append(mask_frame)
                # print("*****mouth_mask***")
                # print(len(mouth_mask))
                mouth_contour.append(mean_mask_frame)
                referece.append(ref_frame)
                can.append(can_frame)
                
            mouth_masks=mouth_masks+mouth_mask
            mouth_contours=mouth_contours+mouth_contour
            refereces=refereces+referece
            cans=cans+can
        return mouth_masks, mouth_contours,refereces,cans
    
    #获取数据集中指定索引位置的数据，调用 make_dataset 方法获取数据后，
    # 将相关图像进行缩放、归一化处理，并转换为 PyTorch 张量，并返回嘴部蒙版图像、嘴部轮廓图像和参考图像。
    def __getitem__(self, index):    #每次都取
        #print("Getitem")
        # mouth_mask, mouth_contour, referece = self.make_dataset(self.folder_path)
        #can_index = random.randint(0, 132)
        #print("can_index",can_index)
        mouth_mask_img = cv2.resize(self.mouth_masks[index], (self.mouth_size, self.mouth_size)) / 255.
        mouth_contour_img = cv2.resize(self.mouth_contours[index], (self.mouth_size, self.mouth_size)) / 255.
        referece_img = cv2.resize(self.refereces[index], (self.mouth_size, self.mouth_size)) / 255.
        can_img=cv2.resize(self.cans[index], (self.mouth_size, self.mouth_size)) / 255.

        mouth_mask_img = torch.from_numpy(mouth_mask_img).permute(2, 0, 1).float().cuda()
        mouth_contour_img = torch.from_numpy(mouth_contour_img).permute(2, 0, 1).float().cuda()
        referece_img = torch.from_numpy(referece_img).permute(2, 0, 1).float().cuda()
        can_img=torch.from_numpy(can_img).permute(2, 0, 1).float().cuda()
        
        return mouth_mask_img, mouth_contour_img, referece_img,can_img
        
    def __len__(self):
        print("*sel**")
        print(self.listlen)
        return self.listlen


















