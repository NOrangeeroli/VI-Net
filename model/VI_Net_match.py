import torch
import torch.nn as nn
import torch.nn.functional as F

from module import SphericalFPN, V_Branch, I_Branch, I_Branch_Pair
from extractor_dino import ViTExtractor
from loss import SigmoidFocalLoss
from smap_utils import Feat2Smap
from torchvision import transforms
from rotation_utils import angle_of_rotation
def plot_pt(pt,index):
    import pyvista as pv
    import matplotlib.pyplot as plt
    cloud = pv.PolyData(pt[index].cpu().numpy())
    mesh = cloud.delaunay_2d()
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white')
    _ = plotter.add_axes(box=True)
    
    
    plotter.show()

def plot_rgb(rgb, index):
    from matplotlib import pyplot as plt
    plt.imshow(rgb[index].reshape(60,60,-1).cpu().numpy(), interpolation='nearest')
    plt.show()

class Net(nn.Module):
    def __init__(self, resolution=64, ds_rate=2):
        super(Net, self).__init__()
        self.res = resolution
        self.ds_rate = ds_rate
        self.ds_res = resolution//ds_rate
        self.extractor = ViTExtractor('dinov2_vits14', 14, device = 'cuda')
        self.extractor_preprocess = transforms.Normalize(mean=self.extractor.mean, std=self.extractor.std)
        self.extractor_layer = 11
        self.extractor_facet = 'token'
        self.extractor_scale = 14
        self.match_sample_num = 128
        self.num_patches = 60

        # data processing
        self.feat2smap = Feat2Smap(self.res)

        self.spherical_fpn = SphericalFPN(ds_rate=self.ds_rate, dim_in1=1, dim_in2=3+384)
        self.v_branch = V_Branch(resolution=self.ds_res, in_dim = 256*2)
        self.i_branch = I_Branch_Pair(resolution=self.ds_res, in_dim = 256*2)
    def rotate_pts_batch(self,pts, rotation):
            pts_shape = pts.shape
            b = pts_shape[0]
            
            return (rotation[:,None,:,:]@pts.reshape(b,-1,3)[:,:,:,None]).squeeze().reshape(pts_shape)

    def extract_feature(self, rgb_raw):
        
      
        
        
        
        
        rgb_raw = rgb_raw.permute(0,3,1,2)
        
        rgb_raw = self.extractor_preprocess(rgb_raw)
        #import pdb;pdb.set_trace()
        
        with torch.no_grad():
            #dino_feature = self.extractor.model(rgb_raw)
        
            dino_feature = self.extractor.extract_descriptors(rgb_raw, layer = self.extractor_layer, facet = self.extractor_facet )
        
        dino_feature = dino_feature.reshape(dino_feature.shape[0],self.num_patches,self.num_patches,-1)
        
        return dino_feature.contiguous() # b x c x h x w
        
    def forward(self, inputs):
        # import pdb;pdb.set_trace()
        rgb1, rgb2 = inputs['rgb'][:,0,:,:], inputs['rgb'][:,1,:,:]
        pts1, pts2 = inputs['pts'][:,0,:,:], inputs['pts'][:,1,:,:]
        b,_,_,_,_ = inputs['rgb_raw'].shape
        rotation_ref = inputs['rotation_ref']
        rgb_raw = inputs['rgb_raw'].reshape(-1,840,840,3)
        
        #pts1 = self.rotate_pts_batch(pts1, rotation_ref)
        #pts2 = self.rotate_pts_batch(pts2, rotation_ref.transpose(1,2))
        pts2 = self.rotate_pts_batch(pts2, rotation_ref.transpose(1,2))
        #import pdb;pdb.set_trace()
        feature = self.extract_feature(rgb_raw).reshape(b*2,3600,-1)
        _,_,num_sample = inputs['choose'].shape
        
        feature = feature[torch.arange(b*2)[:,None], inputs['choose'].reshape(b*2,num_sample),:].reshape(b,2,num_sample,-1)
        feature1, feature2 = feature[:,0], feature[:,1]
        pts_raw = inputs['pts_raw']
        pts_raw = pts_raw.reshape(b*2,3600,-1)[torch.arange(b*2)[:,None], inputs['choose'].reshape(b*2,num_sample),:].reshape(b,2,num_sample,-1)
        ptsf1, ptsf2 = pts_raw[:,0], pts_raw[:,1]

        
        #plot_pt(pts2[0].cpu().numpy())
        dis_map1, rgb_map1 = self.feat2smap(pts1, rgb1)
        dis_map2, rgb_map2 = self.feat2smap(pts2, rgb2)
        _,dino_map1 = self.feat2smap(ptsf1, feature1)
        _,dino_map2 = self.feat2smap(ptsf2, feature2)
        # dis_map = torch.cat([dis_map1, dis_map2], axis = 1)
        # rgb_map = torch.cat([rgb_map1, rgb_map2], axis = 1)
        rgb_map1 = torch.cat([rgb_map1, dino_map1], axis = -3)
        rgb_map2 = torch.cat([rgb_map2, dino_map2], axis = -3)

        # backbone
        x1 = self.spherical_fpn(dis_map1, rgb_map1)
        x2 = self.spherical_fpn(dis_map2, rgb_map2)
        
        x = torch.cat([x1, x2], axis = 1)
        # viewpoint rotation
        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)
        pred_vp_rot = self.v_branch._get_vp_rotation(rho_prob, phi_prob,{})

        # in-plane rotation
        ip_rot = self.i_branch(x1, x2, vp_rot)

        outputs = {
            'pred_rotation': vp_rot @ ip_rot,
            'pred_vp_rotation': pred_vp_rot,
            'pred_ip_rotation': ip_rot,
            'rho_prob': rho_prob,
            'phi_prob': phi_prob
        }
        return outputs


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.l1loss = nn.L1Loss()
        self.sfloss = SigmoidFocalLoss()

    def forward(self, pred, gt):
        rho_prob = pred['rho_prob']
        rho_label = F.one_hot(gt['rho_label'].squeeze(1), num_classes=rho_prob.size(1)).float()
        #import pdb;pdb.set_trace()
        rho_loss = self.sfloss(rho_prob, rho_label).mean()
        pred_rho = torch.max(torch.sigmoid(rho_prob),1)[1]
        rho_acc = (pred_rho.long() == gt['rho_label'].squeeze(1).long()).float().mean() * 100.0

        phi_prob =  pred['phi_prob']
        phi_label = F.one_hot(gt['phi_label'].squeeze(1), num_classes=phi_prob.size(1)).float()
        phi_loss = self.sfloss(phi_prob, phi_label).mean()
        pred_phi = torch.max(torch.sigmoid(phi_prob),1)[1]
        phi_acc = (pred_phi.long() == gt['phi_label'].squeeze(1).long()).float().mean() * 100.0
        
        vp_loss = rho_loss + phi_loss
        ip_loss = self.l1loss(pred['pred_rotation'], gt['rotation_label'])
        residual_angle = angle_of_rotation(pred['pred_rotation'].transpose(1,2) @ gt['rotation_label'])

        vp_residual_angle = angle_of_rotation(pred['pred_vp_rotation'].transpose(1,2) @ gt['vp_rotation_label'])
        ip_residual_angle = angle_of_rotation(pred['pred_ip_rotation'].transpose(1,2) @ gt['ip_rotation_label'])

        loss = self.cfg.vp_weight * vp_loss + ip_loss

        return {
            'loss': loss,
            'vp_loss': vp_loss,
            'ip_loss': ip_loss,
            'rho_acc': rho_acc,
            'phi_acc': phi_acc,
            'residual_angle':residual_angle.mean(),
            'vp_residual_angle':vp_residual_angle.mean(),
            'ip_residual_angle': ip_residual_angle.mean()
        }
