import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from extractor_dino import ViTExtractor
from module import SphericalFPN, V_Branch, I_Branch, I_Branch_Pair
from loss import SigmoidFocalLoss
from smap_utils import Feat2Smap

from rotation_utils import angle_of_rotation

def choose_roi_to_coordinate(choose_roi, rmin, rmax, cmin, cmax):
    assert len(choose_roi.shape) == 2
    #roi_r = rmax - rmin
    roi_c = cmax - cmin
    r_roi = choose_roi//roi_c[:,None]
    c_roi = choose_roi%roi_c[:,None]
    r, c = r_roi + rmin[:,None], c_roi + cmin[:,None]
    
    return torch.stack([r,c],dim = -1)

def coordinate_to_choose(coordinate, h, w):
    #roi_r = rmax - rmin
    #import pdb;pdb.set_trace()
    r,c = coordinate[:,:,0], coordinate[:,:,1]
    
    choose = w * r + c
    return choose


class Net(nn.Module):
    def __init__(self, resolution=64, ds_rate=2):
        super(Net, self).__init__()
        self.res = resolution
        self.ds_rate = ds_rate
        self.ds_res = resolution//ds_rate
        
        
        
        self.extractor = ViTExtractor('dinov2_vitb14', 14, device = 'cuda')
        self.extractor_preprocess = transforms.Normalize(mean=self.extractor.mean, std=self.extractor.std)
        self.extractor_layer = 11
        self.extractor_facet = 'token'
        self.extractor_scale = 16
        # data processing
        self.feat2smap = Feat2Smap(self.res)

        self.spherical_fpn = SphericalFPN(ds_rate=self.ds_rate, dim_in1=1, dim_in2=3*2)
        self.v_branch = V_Branch(resolution=self.ds_res, in_dim = 256*2)
        self.i_branch = I_Branch_Pair(resolution=self.ds_res, in_dim = 256*2)
    def extract_feature(self, rgb_raw):
        
        
        _, image_h, image_w, _ = rgb_raw.shape
        rgb_raw = rgb_raw.permute(0,3,1,2)
        rgb_raw = F.interpolate(rgb_raw, mode = 'bilinear', size = (840,840))
        rgb_raw = self.extractor_preprocess(rgb_raw/255.0)

        with torch.no_grad():
        
            dino_feature = self.extractor.extract_descriptors(rgb_raw, layer = self.extractor_layer, facet = self.extractor_facet )
        dino_feature = dino_feature.reshape(dino_feature.shape[0],60,60,-1).permute(0,3,1,2)
        
        dino_feature = F.interpolate(dino_feature, mode = 'bilinear', size = (int(image_h/self.extractor_scale),int(image_w/self.extractor_scale)))
        
        return dino_feature # b x c x h x w


    def matcher(self, feature1, feature2):
        assert feature1.shape[0] == feature2.shape[0]
        assert feature1.shape[2] == feature2.shape[1]
        
        b, c,h,w = feature2.shape
        b,s,c = feature1.shape
        
        feature2= feature2.permute(0,2,3,1).reshape(b,h*w,c)
        cos_sim = torch.cdist(feature1, feature2)
        match_choose = torch.max(cos_sim, dim = -1)[1]

        
        return match_choose #b x s

    def choose_from_feature(self, feature, rmin, rmax, cmin, cmax, choose):

        coordinate_orig = choose_roi_to_coordinate(choose, rmin, rmax, cmin, cmax)
        coordinate = coordinate_orig//self.extractor_scale
        b, c, h, w = feature.shape
        choose = coordinate_to_choose(coordinate, h, w)
        return feature.reshape(b,c,h*w)[torch.arange(b)[:,None], :, choose] #b x c x sample



    


        
    def forward(self, inputs):
        #import pdb;pdb.set_trace()
        rgb_raw1, rgb_raw2 = inputs['rgb_raw'][:,0,:,:,:], inputs['rgb_raw'][:,1,:,:,:]
        pts_raw1, pts_raw2 = inputs['pts_raw'][:,0,:,:,:], inputs['pts_raw'][:,1,:,:,:]
        choose1, choose2 = inputs['choose'][:,0,:], inputs['choose'][:,1,:]
        rmax1, rmax2 = inputs['rmax'][:,0], inputs['rmax'][:,1]
        rmin1, rmin2 = inputs['rmin'][:,0], inputs['rmin'][:,1]
        cmax1, cmax2 = inputs['cmax'][:,0], inputs['cmax'][:,1]
        cmin1, cmin2 = inputs['cmin'][:,0], inputs['cmin'][:,1]

        mask1, mask2 = inputs['mask'][:,0,:,:], inputs['mask'][:,1,:,:]
        assert mask1.shape == rgb_raw1.shape[:3]
        
        mask1 = mask1[:,None,:,:].float()
        mask2 = mask2[:,None,:,:].float()
        b,_,h,w = mask1.shape
        mask1 = F.interpolate(mask1, mode = 'nearest', size = (h//self.extractor_scale, w//self.extractor_scale))
        mask2 = F.interpolate(mask2, mode = 'nearest', size = (h//self.extractor_scale, w//self.extractor_scale))
        


        dino_feature1 = self.extract_feature(rgb_raw1) * mask1
        dino_feature2 = self.extract_feature(rgb_raw2) * mask2



        dino_feature1_filter1 = self.choose_from_feature(dino_feature1,rmin1, rmax1, cmin1, cmax1, choose1)
        dino_feature1_filter2 = self.choose_from_feature(dino_feature2,rmin2, rmax2, cmin2, cmax2, choose2)

        
        match_choose1 = self.matcher(dino_feature1_filter1, dino_feature2)
        match_choose2 = self.matcher(dino_feature1_filter2, dino_feature1)

        match_coord1 = choose_roi_to_coordinate(match_choose1, 
                                                rmin = torch.IntTensor([0]*b).cuda(), 
                                                cmin = torch.IntTensor([0]*b).cuda(), 
                                                rmax = torch.IntTensor([h//self.extractor_scale]*b).cuda(), 
                                                cmax = torch.IntTensor([w//self.extractor_scale]*b).cuda())
        match_coord2 = choose_roi_to_coordinate(match_choose2, 
                                                rmin = torch.IntTensor([0]*b).cuda(), 
                                                cmin = torch.IntTensor([0]*b).cuda(), 
                                                rmax = torch.IntTensor([h//self.extractor_scale]*b).cuda(), 
                                                cmax = torch.IntTensor([w//self.extractor_scale]*b).cuda())
        match_coord1 = match_coord1 * self.extractor_scale
        match_coord2 = match_coord2 * self.extractor_scale

        match_choose1 = coordinate_to_choose(match_coord1, h, w)
        match_choose2 = coordinate_to_choose(match_coord2, h, w)

        match_pts1 = pts_raw2.reshape(b,h*w,3)[torch.arange(b)[:,None], match_choose1,:]
        match_pts2 = pts_raw1.reshape(b,h*w,3)[torch.arange(b)[:,None], match_choose2,:]

        








        rgb1, rgb2 = inputs['rgb'][:,0,:,:], inputs['rgb'][:,1,:,:]
        pts1, pts2 = inputs['pts'][:,0,:,:], inputs['pts'][:,1,:,:]

        drift1 =  match_pts1 - pts1
        drift2 =  match_pts2 - pts2
        

        dis_map1, rgb_map1 = self.feat2smap(pts1, torch.concatenate([rgb1, drift1], dim = -1))
        dis_map2, rgb_map2 = self.feat2smap(pts2, torch.concatenate([rgb2, drift2], dim = -1))
        # dis_map = torch.cat([dis_map1, dis_map2], axis = 1)
        # rgb_map = torch.cat([rgb_map1, rgb_map2], axis = 1)
        
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
