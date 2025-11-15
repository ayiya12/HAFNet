import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from hafnet.layers.acu import ACU   
from hafnet.layers.msrfu import SKConv as MSRFU  
from hafnet.layers.fsau import FreqSpatialAttention as FSAU

class HARB(nn.Module):  
    """Hybrid Attention-Driven Residual Block"""  
    def __init__(self, channels, cluster_num, filter_threshold,   
                 cluster_source="channel",
                 enable_msrfu=True,       
                 enable_fsau=True):  
        super().__init__()  
        self.enable_msrfu = enable_msrfu  
        self.enable_fsau = enable_fsau
        # ACU layers  
        self.acu1 = ACU(channels, channels, cluster_num=cluster_num,  
                       cluster_source=cluster_source, filter_threshold=filter_threshold)  
        self.acu2 = ACU(channels, channels, cluster_num=cluster_num,  
                       filter_threshold=filter_threshold)  
          
        if enable_msrfu:  
            self.msrfu = MSRFU(channels, num_branches=2)  
        if enable_fsau:  
            self.fsau = FSAU(channels)

        self.act = nn.LeakyReLU(inplace=True)  
        
        self.logger = logging.getLogger(__name__)

        self.forward_count = 0
  
    def forward(self, x, cache_indice=None, cluster_override=None):    
        input_orig = x.clone()  
        
        input_mean = x.mean().item()    
        self.logger.debug(f"Input mean: {input_mean:.6f}")  
        res, idx = self.acu1(x, cache_indice, cluster_override)  
        res = self.act(res)    
        res, _ = self.acu2(res, cache_indice, idx)  
        acu_mean = res.mean().item()    
        self.logger.debug(f"After ACU mean: {acu_mean:.6f}")  
    
        res_before_msrfu = res.clone()  
   
        if self.enable_msrfu:    
            res = self.msrfu(F.leaky_relu(res))  
            self.logger.debug(f"MSRFU applied - Mean: {res.mean().item():.6f}")   
    
        res_before_fsau = res.clone()   
  
        if self.enable_fsau:    
            res = res + self.fsau(res)  
            res = F.leaky_relu(res)    
            self.logger.debug(f"FSAU applied - Mean: {res.mean().item():.6f}")  
  
        x = x + res   
        output_mean = x.mean().item()    
        self.logger.debug(f"Output mean: {output_mean:.6f}")   
        self.forward_count += 1   
    
        return x, idx
            

class HAFNet(nn.Module):  
    """Hybrid Attention Fusion Network"""    
    def __init__(self, spectral_num=8, channels=32, cluster_num=32,     
                 filter_threshold=0.005, enable_msrfu=True,  
                 enable_fsau=True):  
        super().__init__()    
        self.head_conv = nn.Conv2d(spectral_num+1, channels, 3, 1, 1)    
    
        self.rb1 = HARB(channels, cluster_num, filter_threshold,  
                       enable_msrfu=enable_msrfu,    
                       enable_fsau=enable_fsau)   
        self.down1 = ConvDown(channels)    
            
        self.rb2 = HARB(channels*2, cluster_num, filter_threshold,   
                       enable_msrfu=enable_msrfu,    
                       enable_fsau=enable_fsau)  
        self.down2 = ConvDown(channels*2)    
            
        self.rb3 = HARB(channels*4, cluster_num, filter_threshold,    
                       enable_msrfu=enable_msrfu,    
                       enable_fsau=enable_fsau)   
          
        self.up1 = ConvUp(channels*4)    
          
        self.rb4 = HARB(channels*2, cluster_num, filter_threshold,   
                       enable_msrfu=enable_msrfu,    
                       enable_fsau=enable_fsau)  
  
        self.up2 = ConvUp(channels*2)    
        self.rb5 = HARB(channels, cluster_num, filter_threshold, 
                       enable_msrfu=enable_msrfu,    
                       enable_fsau=enable_fsau)  
                                   
        self.tail_conv = nn.Conv2d(channels, spectral_num, 3, 1, 1)  
  
    def forward(self, pan, lms, cache_indice=None):  
        x1 = torch.cat([pan, lms], dim=1)  
        x1 = self.head_conv(x1)  
        x1, idx1 = self.rb1(x1, cache_indice)  
        x2 = self.down1(x1)  
        x2, idx2 = self.rb2(x2, cache_indice)  
        x3 = self.down2(x2)  
        x3, _ = self.rb3(x3, cache_indice)  
        x4 = self.up1(x3, x2)  
        del x2  
        x4, _ = self.rb4(x4, cache_indice, idx2)  
        del idx2  
        x5 = self.up2(x4, x1)  
        del x1  
        x5, _ = self.rb5(x5, cache_indice, idx1)  
        del idx1  
        x5 = self.tail_conv(x5)  
        return lms + x5

class ConvDown(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if dsconv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                          groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels*2, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels*2, 3, 1, 1)
            )

    def forward(self, x):
        return self.conv(x)


class ConvUp(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.acu1 = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2, 0)
        if dsconv:
            self.acu2 = nn.Sequential(
                nn.Conv2d(in_channels//2, in_channels//2, 3, 1,
                          1, groups=in_channels//2, bias=False),
                nn.Conv2d(in_channels//2, in_channels//2, 1, 1, 0)
            )
        else:
            self.acu2 = nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1)

    def forward(self, x, y):
        x = F.leaky_relu(self.acu1(x))
        x = x + y
        x = F.leaky_relu(self.acu2(x))
        return x


