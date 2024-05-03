import cv2
import numpy as np
from util_calculate_psnr_ssim import*
import torch




def read_video_frames(video_path, num_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if num_frames is not None and len(frames) >= num_frames:
            break
    cap.release()
    return frames

def main():
    # Input video paths
    video_path1 = "/home/lcj/Project/yangl/HDTR-Net-main/sample/out_ZYnv_mouth_XZ0013.mp4"
    video_path2 = "/home/lcj/Project/yangl/HDTR-Net-main/test_result/ZYnv_tooth_tooth_230/out_ZYnv_tooth_tooth_230.mp4"
    print(video_path1)
    print(video_path2)

    # Read video frames
    frames1 = read_video_frames(video_path1)
    frames2 = read_video_frames(video_path2)

    # Choose the minimum number of frames
    min_frames = min(len(frames1), len(frames2))
    
    # Calculate PSNR for each frame pair
    psnr_values = []
    for i in range(min_frames):
        frame1 = frames1[i]
        frame2 = frames2[i]
        psnr = calculate_psnr(frame1, frame2, crop_border=0, input_order='HWC', test_y_channel=False)
        psnr_values.append(psnr)

    # Calculate average PSNR
    avg_psnr = np.mean(psnr_values)
    print("Average PSNR:", avg_psnr)

    # Calculate PSNRB for each frame pair
    psnrb_values = []
    for i in range(min_frames):
        frame1 = frames1[i]
        frame2 = frames2[i]
        psnrb = calculate_psnrb(frame1, frame2, crop_border=0, input_order='HWC', test_y_channel=False)
        psnrb_values.append(psnrb)

    # Calculate average PSNRB
    avg_psnrb = np.mean(psnrb_values)
    print("Average PSNRB:", avg_psnrb)
    
    # Calculate SSIM for each frame pair
    ssim_values = []
    for i in range(min_frames):
        frame1 = frames1[i]
        frame2 = frames2[i]
        ssim = calculate_ssim(frame1, frame2, crop_border=0, input_order='HWC', test_y_channel=False)
        ssim_values.append(ssim)

    # Calculate average SSIM
    avg_ssim = np.mean(ssim_values)
    print("Average SSIM:", avg_ssim)

if __name__ == "__main__":
    main()