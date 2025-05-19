'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:00:38
Description: 
'''

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
from util.metrics import DeepfakeMetrics

import time
import os
import torch
import platform
import psutil
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# detransformer = transforms.Compose([
#         transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
#         transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
#     ])


if __name__ == '__main__':
    opt = TestOptions().parse()
    metrics_calculator = DeepfakeMetrics()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    model = create_model(opt)
    model.eval()


    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)
    with torch.no_grad():
        pic_a = opt.pic_a_path
        # img_a = Image.open(pic_a).convert('RGB')
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # pic_b = opt.pic_b_path
        # img_b_whole = cv2.imread(pic_b)
        # img_b_align_crop, b_mat = app.get(img_b_whole,crop_size)
        # img_b_align_crop_pil = Image.fromarray(cv2.cvtColor(img_b_align_crop,cv2.COLOR_BGR2RGB)) 
        # img_b = transformer(img_b_align_crop_pil)
        # img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        # img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)


        # My code
        start_time = time.time()
        # Sistem bilgisi
        cpu_info = platform.processor()
        ram_total = round(psutil.virtual_memory().total / (1024 ** 3), 2)  # GB
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
        # Video çözünürlüğü (cv2 ile çekiyoruz)
        video_capture = cv2.VideoCapture(opt.video_path)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        # Frame sayacı ve yüz algılama sayacı (global değişken)
        global_frame_counter = 0
        global_face_counter = 0
        # My code

        global_frame_counter, global_face_counter =  video_swap(opt.video_path, latend_id, model, app, opt.output_path,temp_results_dir=opt.temp_path,\
            no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size)
        
    # My code
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    fps = round(global_frame_counter / total_time, 2) if total_time > 0 else 0

    output_size_mb = round(os.path.getsize(opt.output_path) / (1024 * 1024), 2)

    # Metrik hesaplama
    video_capture = cv2.VideoCapture(opt.video_path)
    ret, original_frame = video_capture.read()
    video_capture.release()
    
    output_capture = cv2.VideoCapture(opt.output_path)
    ret, swapped_frame = output_capture.read()
    output_capture.release()
    
    if original_frame is not None and swapped_frame is not None:
        metrics = metrics_calculator.calculate_all_metrics(original_frame, swapped_frame)
    else:
        metrics = {'ssim': 0, 'psnr': 0, 'lpips': 0, 'face_recognition_accuracy': 0}

    # Log klasörü ve dosyası
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/simswap_metrics_log.csv"
    write_header = not os.path.exists(log_path)

    # Metric headers and descriptions
    metric_headers = [
        "Timestamp",
        "Input Video",
        "Output Video",
        "Resolution",
        "Crop Size",
        "Total Frames",
        "Detected Faces",
        "Face Detection Rate (%)",
        "Processing Time (s)",
        "Original FPS",
        "Processing FPS",
        "Output Size (MB)",
        "GPU",
        "CPU",
        "RAM (GB)",
        "SSIM (0-1)",
        "PSNR (dB)",
        "LPIPS (0-1)",
        "Face Recognition Accuracy (0-1)"
    ]

    with open(log_path, "a", encoding='utf-8') as f:
        if write_header:
            f.write(",".join(metric_headers) + "\n")
        
        # Format metric values
        metrics_data = [
            f'"{datetime.now()}"',
            f'"{opt.video_path}"',
            f'"{opt.output_path}"',
            f'"{frame_width}x{frame_height}"',
            str(opt.crop_size),
            str(global_frame_counter),
            str(global_face_counter),
            f"{round(global_face_counter/global_frame_counter*100, 2)}",
            f"{total_time}",
            f"{original_fps:.2f}",
            f"{fps}",
            f"{output_size_mb}",
            f'"{gpu_name}"',
            f'"{cpu_info}"',
            str(ram_total),
            f"{metrics['ssim']:.4f}",
            f"{metrics['psnr']:.2f}",
            f"{metrics['lpips']:.4f}",
            f"{metrics['face_recognition_accuracy']:.4f}"
        ]
        
        f.write(",".join(metrics_data) + "\n")

    # Print summary to console
    print("\n=== Metric Summary ===")
    print(f"SSIM: {metrics['ssim']:.4f} (closer to 1 is better)")
    print(f"PSNR: {metrics['psnr']:.2f} dB (higher is better)")
    print(f"LPIPS: {metrics['lpips']:.4f} (closer to 0 is better)")
    print(f"Face Recognition Accuracy: {metrics['face_recognition_accuracy']:.4f} (closer to 1 is better)")
    print(f"Processing Time: {total_time:.2f} seconds")
    print(f"Original Video FPS: {original_fps:.2f}")
    print(f"Processing FPS: {fps:.2f} (frames processed per second)")
    print("===================\n")

    # Create visualizations
    os.makedirs("logs/visualizations", exist_ok=True)
    
    # Get source and target names for file naming
    source_name = os.path.splitext(os.path.basename(opt.pic_a_path))[0]
    target_name = os.path.splitext(os.path.basename(opt.video_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Read the CSV file
    df = pd.read_csv(log_path)
    
    # Create a bar plot for quality metrics
    plt.figure(figsize=(12, 6))
    quality_metrics = ['SSIM (0-1)', 'PSNR (dB)', 'LPIPS (0-1)', 'Face Recognition Accuracy (0-1)']
    values = [metrics['ssim'], metrics['psnr'], metrics['lpips'], metrics['face_recognition_accuracy']]
    
    plt.bar(quality_metrics, values)
    plt.title(f'Quality Metrics - {source_name} to {target_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"logs/visualizations/{source_name}_to_{target_name}_{timestamp}_quality_metrics.png")
    plt.close()

    # Create a performance metrics plot
    plt.figure(figsize=(10, 6))
    performance_metrics = ['Processing Time (s)', 'FPS', 'Face Detection Rate (%)']
    values = [total_time, fps, round(global_face_counter/global_frame_counter*100, 2)]
    
    plt.bar(performance_metrics, values)
    plt.title(f'Performance Metrics - {source_name} to {target_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"logs/visualizations/{source_name}_to_{target_name}_{timestamp}_performance_metrics.png")
    plt.close()

    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_columns = ['SSIM (0-1)', 'PSNR (dB)', 'LPIPS (0-1)', 'Face Recognition Accuracy (0-1)', 
                      'Processing Time (s)', 'Original FPS', 'Processing FPS', 'Face Detection Rate (%)']
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Metrics Correlation Heatmap - {source_name} to {target_name}')
    plt.tight_layout()
    plt.savefig(f"logs/visualizations/{source_name}_to_{target_name}_{timestamp}_correlation_heatmap.png")
    plt.close()

