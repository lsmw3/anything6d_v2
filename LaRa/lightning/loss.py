import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM
from torch.nn import functional as F

from torch.cuda.amp import autocast
#from third_party.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pytorch3d.loss import chamfer_distance

class Losses(nn.Module):
    def __init__(self):
        super(Losses, self).__init__()

        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        #self.chamferdistance = chamfer_distance()

    def cos_loss(self, output, gt, mask, thrsh=0, weight=1):
        cos = torch.sum(output * gt * mask * weight, -1)
        masked_cos = cos[mask.squeeze(-1) == 1]
        return (1 - masked_cos[masked_cos < np.cos(thrsh)]).mean()

    def forward(self, batch, output, iter, start_triplane):

        scalar_stats = {}
        loss = 0

        B,V,H,W = batch['tar_rgb'].shape[:-1]
        tar_rgb = batch['tar_rgb'].permute(0,2,1,3,4).reshape(B,H,V*W,3)
        
        # B,V,H,W = batch['mask'].shape
        mask = batch['mask'].permute(0,2,1,3).reshape(B,H,V*W).unsqueeze(-1)
            
        if 'image' in output:

            for prex in ['','_fine']:
                
                if prex=='_fine' and f'acc_map{prex}' not in output:
                    continue

                if start_triplane:
                    color_loss_all = (output[f'image{prex}']-tar_rgb)**2
                    # loss += color_loss_all[mask.expand(-1, -1, -1, 3) == 1].mean()
                    loss += color_loss_all.mean()

                    psnr = -10. * torch.log(color_loss_all.detach().mean()) / \
                        torch.log(torch.Tensor([10.]).to(color_loss_all.device))
                    scalar_stats.update({f'mse{prex}': color_loss_all.mean().detach()})
                    scalar_stats.update({f'psnr{prex}': psnr})


                    with autocast(enabled=False): 
                        ssim_val = self.ssim(output[f'image{prex}'].permute(0,3,1,2), tar_rgb.permute(0,3,1,2))
                        scalar_stats.update({f'ssim{prex}': ssim_val.detach()})
                        # if with_fine:
                        #     loss += 0.02 * (1-ssim_val)
                        # else:
                        #     loss += 0.005 * (1-ssim_val)
                        loss += 0.02 * (1-ssim_val)
                    
                    if f'rend_dist{prex}' in output and prex!='_fine': #and iter>1000:
                        distortion = output[f"rend_dist{prex}"].mean()
                        scalar_stats.update({f'distortion{prex}': distortion.detach()})
                        loss += distortion*1000
                        
                        rend_normal_world  = output[f'rend_normal{prex}']
                        depth_normal = output[f'depth_normal{prex}']
                        gt_normal_world = batch['tar_nrm'] if 'tar_nrm' in batch else None
                        acc_map = output[f'acc_map{prex}'].detach()

                        if gt_normal_world is not None:
                            loss_surface = self.cos_loss(rend_normal_world, gt_normal_world, mask)
                            scalar_stats.update({f'normal{prex}': loss_surface.detach()})
                            # if with_fine:
                            #     loss += loss_surface*0.2
                            # else:
                            #     loss += loss_surface*0.02
                            loss += loss_surface * 0.2 # / (loss_surface / loss_cd).detach()

                        normal_error = ((1 - (rend_normal_world * depth_normal).sum(dim=-1))*acc_map).mean()
                        scalar_stats.update({f'depth_norm{prex}': normal_error.detach()})
                        # if with_fine:
                        #     loss += normal_error*0.2
                        # else:
                        #     loss += normal_error*0.02
                        loss += normal_error * 0.2 # / (normal_error/ loss_cd).detach()
                else:
                    if 'pred_pc' in output:
                        loss_cd,loss_cdn = chamfer_distance(output['pred_pc'], output['gt_pc'])
                        loss = loss_cd
                        # loss += loss_cd
                        scalar_stats.update({f'chamferdist': loss_cd.detach()})
                    else:
                        raise NotImplementedError("There's no predicted point cloud in the output!!!")
     
        return loss, scalar_stats