import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from insightface.app import FaceAnalysis

class DeepfakeMetrics:
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='alex').cuda()
        self.face_analyzer = FaceAnalysis(name='antelope', root='./insightface_func/models')
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    
    def calculate_ssim(self, img1, img2):
        """Yapısal Benzerlik İndeksi hesaplama"""
        # Görüntüleri BGR'den RGB'ye dönüştür
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Görüntü boyutlarını kontrol et
        min_dim = min(img1.shape[0], img1.shape[1])
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
            
        return ssim(img1, img2, channel_axis=2, win_size=win_size)
    
    def calculate_psnr(self, img1, img2):
        """Piksel Benzerlik İndeksi hesaplama"""
        # Görüntüleri BGR'den RGB'ye dönüştür
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        return psnr(img1, img2)
    
    def calculate_lpips(self, img1, img2):
        """LPIPS metrik hesaplama"""
        # Görüntüleri BGR'den RGB'ye dönüştür
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return self.loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda()).item()
    
    def calculate_face_recognition_accuracy(self, original_img, swapped_img):
        """Yüz tanıma doğruluğu hesaplama"""
        original_faces = self.face_analyzer.get(original_img)
        swapped_faces = self.face_analyzer.get(swapped_img)
        
        if len(original_faces) == 0 or len(swapped_faces) == 0:
            return 0.0
            
        original_embedding = original_faces[0].embedding
        swapped_embedding = swapped_faces[0].embedding
        
        similarity = np.dot(original_embedding, swapped_embedding) / (
            np.linalg.norm(original_embedding) * np.linalg.norm(swapped_embedding)
        )
        return float(similarity)
    
    def calculate_all_metrics(self, original_img, swapped_img):
        """Tüm metrikleri hesapla ve sonuçları döndür"""
        try:
            metrics = {
                'ssim': self.calculate_ssim(original_img, swapped_img),
                'psnr': self.calculate_psnr(original_img, swapped_img),
                'lpips': self.calculate_lpips(original_img, swapped_img),
                'face_recognition_accuracy': self.calculate_face_recognition_accuracy(original_img, swapped_img)
            }
        except Exception as e:
            print(f"Metrik hesaplama hatası: {str(e)}")
            metrics = {
                'ssim': 0,
                'psnr': 0,
                'lpips': 0,
                'face_recognition_accuracy': 0
            }
        return metrics 