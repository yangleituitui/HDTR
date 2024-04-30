from builtins import int, len, print, range, str
import os, cv2
import argparse
import face_alignment
import pickle, time
import shutil

from os.path import join
from tqdm import tqdm
import torch.optim as optim
import logging

import torch
import numpy as np 
from networks.models import *
from networks.discriminator import *
from dataset.inference_dataset_train import InferenceDataset
from torch.utils.data import DataLoader
import os
import json
from tensorboardX import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


#从视频中提取图像帧
split_image_cmd = 'ffmpeg -hwaccel cuvid -hide_banner -loglevel quiet -y -i {} -r 25 {}/%05d.png'
#从视频中提取音频
split_wav_cmd = 'ffmpeg -hwaccel cuvid -hide_banner -loglevel quiet -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'
#提取三维人脸关键点
fa_3d = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cuda:0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter(log_dir='/home/lcj/Project/yangl/HDTR-Net-main/logs/test')
logging.basicConfig(filename='/home/lcj/Project/yangl/HDTR-Net-main/loging/test/test_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#初始化生成器
def load_model(args):
    Model = Generator(args)
    # print("Load checkpoint from: {}".format(args.model_path))
    # checkpoint = torch.load(args.model_path)
    # s = checkpoint["state_dict"]   #加载模型参数
    # new_s = {}              #创建一个新的字典 new_s，用于存储处理后的 state_dict
    # for k, v in s.items():
    #     new_s[k.replace('module.', '')] = v    #不是在多GPU上运行 可以去掉前缀

    # Model.load_state_dict(new_s)     #处理后的 state_dict 加载到模型 Model 中
    # Model = Model.cuda().eval()     #将模型转移到 CUDA 设备上并设置为 evaluation 模式。
    return Model

def jason_file(train_dataloader,json_filename):
    data_list = []
    for mouth_mask, mouth_contour, referece,train_can in train_dataloader:
        batch_data = {
        "mouth_mask": mouth_mask.tolist(),
        "mouth_contour": mouth_contour.tolist(),
        "reference":referece.tolist(),
        "can": train_can.tolist()
    }
    data_list.append(batch_data)
    
    # 将数据列表保存为 JSON 格式的字符串
    json_data = json.dumps(data_list)
    # 将 JSON 数据写入文件
    #json_filename = '/home/lcj/Project/yangl/HDTR-Net-main/data/train_data.json'
    with open(json_filename, 'w') as json_file:
        json_file.write(json_data)
    print(f"训练集数据已保存至 {json_filename}")



# 实例化三个损失函数
GAN_Loss=GANLoss(use_lsgan=True).to(device)
Reconstruction_Loss=reconstructionLoss().to(device)
Perception_Loss=perceptionLoss()
## 设置损失的权重
gan_weight = 0.3
perception_weight = 0.2
reconstruction_weight = 0.5

#训练模型

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def validate_model(generator, discriminator,GAN_Loss,val_dataloader,Perception_Loss, Reconstruction_Loss, gan_weight, perception_weight, reconstruction_weight,epoch):
    generator.eval()
    discriminator.eval()
    num_batches = len(val_dataloader)
    total_generator_loss=0
    print("验证集计算***")
    with torch.no_grad():
        for valid_mouth_mask, valid_mouth_contour, valid_reference,val_can in val_dataloader:
            real_ground_truth = valid_reference.to(device)
            valid_mouth_contour=valid_mouth_contour.to(device)
            valid_reference=valid_reference.to(device)
            generator.to(device)
            predict_data = generator(valid_mouth_mask, valid_mouth_contour, val_can)
            

            gan_loss = GAN_Loss(discriminator(predict_data),target_is_real=True)  #判别器的分数
            perception_loss = Perception_Loss.calculatePerceptionLoss(predict_data, real_ground_truth)
            reconstruction_loss = Reconstruction_Loss.calculateReconstructionLoss(predict_data, real_ground_truth)
            generator_loss =gan_weight*gan_loss+perception_weight * perception_loss + reconstruction_weight * reconstruction_loss
            
            total_generator_loss += generator_loss
        
        avg_generator_loss = total_generator_loss / num_batches
        writer.add_scalar('Val_Generator_Loss', avg_generator_loss, epoch+1)
        logging.info(f'Epoch {epoch+1},Val_Discriminator_Loss: {avg_discriminator_loss}')
        print(f'Epoch [{epoch+1},Val_Generator_Loss: {avg_generator_loss}')
        
            



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouth_size", type=int, default=96, help='Size of the crop mouth region image')
    parser.add_argument("--test_batch_size", type=int, default=8, help='Size of the test batch')
    parser.add_argument("--dis_save_dir", type=str, default="./checkpoint/dis_test")
    parser.add_argument("--gen_save_dir", type=str, default="./checkpoint/gen_test")
    parser.add_argument("--test_workers", type=int, default=0, help='Number of workers to run the test')
    parser.add_argument("--mask_channel", type=int, default=6, help='Channel of the mouth mask and mouth contour')
    parser.add_argument("--ref_channel", type=int, default=3, help='Channel of the reference channel')
    parser.add_argument("--model_path", type=str, default="/home/lcj/Project/yangl/HDTR-Net-main/checkpoint/lip-clarity-model.pth", help='Channel of the reference channel')
    
    parser.add_argument("--train_dataset_path", type=str, default="/home/lcj/Project/yangl/HDTR-Net-main/train")
    parser.add_argument("--val_dataset_path", type=str, default="/home/lcj/Project/yangl/HDTR-Net-main/val")
    
    #parser.add_argument("--train_data_json_path", type=str, default="/home/lcj/Project/yangl/HDTR-Net-main/data/1500/train_data.json")
    #parser.add_argument("--val_data_json_path", type=str, default="/home/lcj/Project/yangl/HDTR-Net-main/data/1500/val_data.json")
    args = parser.parse_args()
    
    generator = load_model(args)
    #判别器
    discriminator = Mouth_disc_qual().to(device)

    #定义优化器
    gen_rate=0.0001
    dis_rate=0.00001
    gen_optimizer = optim.Adam(generator.parameters(), lr= gen_rate)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr= dis_rate)
    logging.info(f'gen_rate {gen_rate}, dis_rate: {dis_rate}')
    
    
    #加载训练集数据
    print("加载训练集数据")
    train_dataset_path=args.train_dataset_path
    train_dataset = InferenceDataset(train_dataset_path,args.mouth_size)
    #train_dataset.make_dataset(train_dataset_path,args.mouth_size)
    train_dataloader = DataLoader(train_dataset,
                            batch_size=args.test_batch_size,
                            shuffle=True,
                            num_workers=args.test_workers,
                            drop_last=False)
    #jason_file(train_dataloader,args.train_data_json_path)
    
    #加载测试集数据
    print("加载验证集数据")
    val_dataset_path=args.val_dataset_path
    val_dataset = InferenceDataset(val_dataset_path,args.mouth_size)
    #train_dataset.make_dataset(train_dataset_path,args.mouth_size)
    val_dataloader = DataLoader(val_dataset,
                            batch_size=args.test_batch_size,
                            shuffle=True,
                            num_workers=args.test_workers,
                            drop_last=False)
    #jason_file(val_dataloader,args.val_data_json_path)
    
    num_epochs = 1000
    
        
    #判别器训练
    for epoch in range(num_epochs):
        print("***epoch{}:".format(epoch))
        num_baches=len(train_dataloader)
        print("num_batches:",num_baches)
        total_discriminator_loss=0
        total_generator_loss=0
        for train_mouth_mask, train_mouth_contour, train_referece, train_can in train_dataloader:
            real_ground_truth=train_referece.to(device)
            train_mouth_mask = train_mouth_mask.to(device)
            train_mouth_contour = train_mouth_contour.to(device)
            train_can = train_can.to(device)
            generator=generator.to(device)
            
            requires_grad(generator, False)
            requires_grad(discriminator, True)
            dis_optimizer.zero_grad()   
            # print("train_can")
            # print(train_can)
            # print("train_referece")
            # print(train_referece)
            
            predict_data = generator(train_mouth_mask, train_mouth_contour,train_can).to(device)
            # print("predict_data")
            # print(predict_data)
            #计算判别器损失
            #print("计算判别器损失")
            real_gt_discriminator_loss = GAN_Loss(discriminator(real_ground_truth), target_is_real=True)
            fake_pd_discriminator_loss = GAN_Loss(discriminator(predict_data), target_is_real=False)
            discriminator_loss = gan_weight*0.5*(real_gt_discriminator_loss +fake_pd_discriminator_loss)
            print("discriminator:",real_gt_discriminator_loss,fake_pd_discriminator_loss)
            total_discriminator_loss+=discriminator_loss
            #判别器优化器反向传播和更新参数
            
            discriminator_loss.backward()
            dis_optimizer.step()
            #print("计算判别器损失done")
            
            requires_grad(generator, True)
            requires_grad(discriminator, False)
            gen_optimizer.zero_grad()
            #计算生成器损失
            #print("计算生成器损失")
            predict_data = generator(train_mouth_mask, train_mouth_contour, train_can)
            perception_Loss=Perception_Loss.calculatePerceptionLoss(predict_data, real_ground_truth)
            reconstruction_Loss=Reconstruction_Loss.calculateReconstructionLoss(predict_data, real_ground_truth)
            gan_loss=GAN_Loss(discriminator(predict_data),target_is_real=True)
            
            generator_loss =gan_weight*gan_loss+perception_weight*perception_Loss+reconstruction_weight*reconstruction_Loss
            total_generator_loss+=generator_loss
            #生成器优化器反向传播
            
            generator_loss.backward()
            gen_optimizer.step()
            #print("计算生成器损失done")

        avg_generator_loss=total_generator_loss/num_baches
        avg_discriminator_loss=total_discriminator_loss/num_baches
        print(f'Epoch [{epoch+1}/{num_epochs}], Discriminator_Loss: {avg_discriminator_loss},Generator_Loss: {avg_generator_loss}')  
        writer.add_scalar('Discriminator_Loss', avg_discriminator_loss, epoch+1)  
        writer.add_scalar('Generator_Loss', avg_generator_loss, epoch+1)
        logging.info(f'Epoch {epoch+1}, Train_Loss: {avg_generator_loss}, Discriminator_Loss: {avg_discriminator_loss}')
        
        discriminator_weights_file = os.path.join(args.dis_save_dir, f'discriminator_weights_epoch{epoch+1}.pth')
        generator_weights_file = os.path.join(args.gen_save_dir, f'generator_weights_epoch{epoch+1}.pth')
        torch.save(discriminator.state_dict(), discriminator_weights_file)
        torch.save(generator.state_dict(), generator_weights_file)
        if (epoch+1)%5==0:
            #print(f'Epoch [{epoch+1}/{num_epochs}], Discriminator Loss: {discriminator_loss.item()},Generator Loss: {generator_loss.item()}')
            validate_model(generator, discriminator,GAN_Loss,val_dataloader,Perception_Loss, Reconstruction_Loss, gan_weight, perception_weight, reconstruction_weight,epoch)
        
    
    
    
    
    
