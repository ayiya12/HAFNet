import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class FreqSpatialAttention(nn.Module):  
    """Frequency-Spatial Attention Unit (FSAU) from HAFNet"""  
    def __init__(self, channels):  
        super().__init__()   
        self.freq_conv = nn.Sequential(  
            nn.Conv2d(channels * 2, channels, 1), 
            nn.ReLU(inplace=True),  
            nn.Conv2d(channels, channels, 1)  
        )  
  
        self.spatial_conv = nn.Sequential(  
            nn.Conv2d(channels, channels, 3, 1, 1),   
            nn.ReLU(inplace=True),  
            nn.Conv2d(channels, channels, 1)  
        )  
            
        self.fusion_weight = nn.Sequential(  
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(channels * 2, 2, 1),  
            nn.Softmax(dim=1)  
        )  
          
    def forward(self, x):  
        B, C, H, W = x.shape  
        x_fft = torch.fft.rfft2(x, norm='ortho')  
        x_fft_real = x_fft.real  
        x_fft_imag = x_fft.imag  
           
        x_fft_cat = torch.cat([x_fft_real, x_fft_imag], dim=1)  
        freq_feat = self.freq_conv(x_fft_cat)  
          
        freq_feat = torch.fft.irfft2(  
            torch.complex(freq_feat, torch.zeros_like(freq_feat)),   
            s=(H, W), norm='ortho'  
        )  
 
        spatial_feat = self.spatial_conv(x)  
 
        fusion_input = torch.cat([freq_feat, spatial_feat], dim=1)  
        weights = self.fusion_weight(fusion_input)  
          
        out = weights[:, 0:1] * freq_feat + weights[:, 1:2] * spatial_feat  
        return out