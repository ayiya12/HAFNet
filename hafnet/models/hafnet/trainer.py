import torch  
import torch.nn as nn  
import os  
from .model import HAFNet  
from hafnet.util.trainer import SimplePanTrainer  
from hafnet.layers.kmeans import reset_cache, KMeansCacheScheduler  

class HAFNetTrainer(SimplePanTrainer):  
    def _create_model(self, cfg):  
        self.criterion = nn.L1Loss().to(self.dev)  
        self.model = HAFNet(  
            spectral_num=cfg['spectral_num'],  
            channels=cfg.get('channels', 32),  
            cluster_num=cfg.get('cluster_num', 32),  
            filter_threshold=cfg.get('filter_threshold', 0.005),
            enable_msrfu=cfg.get('enable_msrfu', True),  
            enable_fsau=cfg.get('enable_fsau', True) 
        ).to(self.dev)  
          
        self.optimizer = torch.optim.Adam(  
            self.model.parameters(),   
            lr=cfg['learning_rate']  
        )  
        self.scheduler = torch.optim.lr_scheduler.StepLR(  
            self.optimizer,   
            step_size=cfg.get('lr_step_size', 200)  
        )  
          
        if 'kmeans_cache_update' in cfg:  
            self.km_scheduler = KMeansCacheScheduler(cfg['kmeans_cache_update'])  
    
    def _compute_val_metrics(self, sr, gt):   
        sam_value = self._compute_sam(sr, gt)  
        return {'SAM': sam_value}  
      
    def _compute_sam(self, sr, gt):  

        sr_flat = sr.permute(0, 2, 3, 1).reshape(-1, sr.shape[1])  
        gt_flat = gt.permute(0, 2, 3, 1).reshape(-1, gt.shape[1])  
  
        dot_product = (sr_flat * gt_flat).sum(dim=1)  
        sr_norm = torch.norm(sr_flat, dim=1)  
        gt_norm = torch.norm(gt_flat, dim=1)  
   
        cos_theta = dot_product / (sr_norm * gt_norm + 1e-8)  
        cos_theta = torch.clamp(cos_theta, -1, 1)  
          
        sam = torch.acos(cos_theta) * 180 / 3.14159265359  
          
        return sam.mean().item()  


    def _on_train_start(self):  
        if hasattr(self, 'km_scheduler'):  
            reset_cache(len(self.train_dataset))  
        self.best_sam = float('inf')  
        self.patience_counter = 0  
        self.patience = self.cfg.get('early_stopping_patience', 50)  
        self.logger.info(f"Early stopping patience: {self.patience}") 
    
    def _on_val_end(self, epoch, val_loss, val_metrics=None):   
        if val_metrics is None or 'SAM' not in val_metrics:  
            self.logger.warning("SAM metric not available, skipping model selection")  
            return False  
          
        current_sam = val_metrics['SAM']  
          
        from torch.utils.tensorboard import SummaryWriter  
        writer_dir = self.out_dir  
        writer = SummaryWriter(log_dir=writer_dir)  
        writer.add_scalar('val/SAM', current_sam, epoch)  
        writer.close()  
          
        if current_sam < self.best_sam:  
            self.best_sam = current_sam  
            self.patience_counter = 0  
            best_path = os.path.join(self.out_dir, 'weights/best.pth')  
            torch.save(self.model.state_dict(), best_path)  
            self.logger.info(f"Epoch {epoch}: New best model saved (SAM: {current_sam:.4f})")  
        else:  
            self.patience_counter += 1  
            self.logger.info(f"Epoch {epoch}: SAM: {current_sam:.4f} (best: {self.best_sam:.4f}, patience: {self.patience_counter}/{self.patience})")  
              
            if self.patience_counter >= self.patience:  
                self.logger.info(f"Early stopping triggered at epoch {epoch}")  
                return True    
          
        return False

      
    def _on_epoch_start(self, epoch):  
        if hasattr(self, 'km_scheduler'):  
            self.km_scheduler.step()  
      
    def forward(self, batch):  
        if "index" in batch:  
            return self.model(batch['pan'].to(self.dev), batch['lms'].to(self.dev), batch['index'].to(self.dev))  
        else:  
            return self.model(batch['pan'].to(self.dev), batch['lms'].to(self.dev))