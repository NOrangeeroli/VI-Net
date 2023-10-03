import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from extractor_dino import ViTExtractor
from module import SphericalFPN, V_Branch, I_Branch, I_Branch_Pair
from loss import SigmoidFocalLoss
from smap_utils import Feat2Smap

from rotation_utils import angle_of_rotation

from time import time
def plot_pt(pt,index):
    import pyvista as pv
    import matplotlib.pyplot as plt
    cloud = pv.PolyData(pt[index].cpu().numpy())
    mesh = cloud.delaunay_2d()
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white')
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
        
        self.patch_size = 14
        self.stride = 14
        self.img_size = 840
        self.num_patches = 60
        



        
        self.extractor = ViTExtractor('dinov2_vits14', self.patch_size, device = 'cuda')
        self.extractor_preprocess = transforms.Normalize(mean=self.extractor.mean, std=self.extractor.std)
        self.extractor_layer = 11
        self.extractor_facet = 'token'
        self.extractor_scale = self.img_size//self.num_patches
        

        self.match_sample_num = 128
        
        # data processing
        self.feat2smap = Feat2Smap(self.res)

        self.spherical_fpn = SphericalFPN(ds_rate=self.ds_rate, dim_in1=1, dim_in2=3)
        self.v_branch = V_Branch(resolution=self.ds_res, in_dim = 256*2)
        self.i_branch = I_Branch_Pair(resolution=self.ds_res, in_dim = 256*2)
   
    def rotate_pts_batch(self,pts, rotation):
            pts_shape = pts.shape
            b = pts_shape[0]

            return (rotation[:,None,:,:]@pts.reshape(b,-1,3)[:,:,:,None]).squeeze().reshape(pts_shape)

    def coordinate_to_choose(self, coordinate, h, w):
        r,c = coordinate[:,:,0], coordinate[:,:,1]
        
        choose = w * r + c
        return choose
    def extract_feature(self, rgb_raw):
        
      
        
        _, image_h, image_w, _ = rgb_raw.shape
        assert image_h == image_w == self.img_size
        
        rgb_raw = rgb_raw.permute(0,3,1,2)
        
        rgb_raw = self.extractor_preprocess(rgb_raw)
        
        
        with torch.no_grad():
        
            dino_feature = self.extractor.extract_descriptors(rgb_raw, layer = self.extractor_layer, facet = self.extractor_facet )
        
        dino_feature = dino_feature.reshape(dino_feature.shape[0],self.num_patches,self.num_patches,-1).permute(0,3,1,2)
        
        return dino_feature.contiguous() # b x c x h x w


    def matcher(self, feature1, feature2, mask1, mask2, choose_backup1, choose_backup2):
        #feature: b x h x w x c
        #mask: b x h x w x 1
        assert feature1.shape == feature2.shape
        assert mask1.shape == mask2.shape
        assert mask1.shape[:3] == feature1.shape[:3]
        
        
        b,h,w,c= feature2.shape
        feature1 = feature1 / feature1.norm(dim=-1, keepdim=True)
        feature2 = feature2 / feature2.norm(dim=-1, keepdim=True)
        feature1= torch.where(mask1>0,feature1, torch.nan ).reshape(b,h*w,c)
        feature2= torch.where(mask2>0,feature2, torch.nan ).reshape(b,h*w,c)
        

        cos_sim = torch.cdist(feature1, feature2) # b x hw x hw
        # import pdb;pdb.set_trace()
        
        cos_sim = torch.where(cos_sim.isnan().logical_not(), cos_sim, 10000000 )
        match1 = torch.argmin(cos_sim, dim = -1)
        match2 = torch.argmin(cos_sim, dim = -2)

        match1_masked = torch.where(mask1.reshape(b,h*w,1).squeeze()>0,match1, torch.nan )
        match2_masked = torch.where(mask2.reshape(b,h*w,1).squeeze()>0,match2, torch.nan )
        

        # import pdb;pdb.set_trace()
        # match31 = torch.topk(-cos_sim, k = 3,dim = -1)[1]
        # match32 = torch.topk(-cos_sim, k = 3,dim = -2)[1]

        match_cyclic1 = torch.gather(match2_masked, dim=-1, index=match1)
        match_cyclic2 = torch.gather(match1_masked, dim=-1, index=match2)
        
        

        coord_cyclic1 = self.choose_to_coordinate(match_cyclic1, h,w)
        coord_cyclic2 = self.choose_to_coordinate(match_cyclic2, h,w)
        coord_cyclic1 = torch.where(mask1.reshape(b,h*w,1)>0, 
                                    coord_cyclic1, torch.nan )
        coord_cyclic2 = torch.where(mask2.reshape(b,h*w,1)>0, 
                                    coord_cyclic2, torch.nan )

        x = torch.tensor(range(h))
        y = torch.tensor(range(w))
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        coord_orig = torch.stack([grid_x, grid_y], dim = -1).reshape(1,h*w, 2).float().cuda()

        coord_diff1 = (coord_cyclic1 - coord_orig).norm(dim = -1)
        coord_diff2 = (coord_cyclic2 - coord_orig).norm(dim = -1)

        coord_diff1 = torch.nan_to_num(coord_diff1, nan=10000)
        coord_diff2 = torch.nan_to_num(coord_diff2, nan=10000)

        choose1 = torch.topk(-coord_diff1, k = self.match_sample_num, dim = -1)[1]
        choose1 = torch.where((mask1.reshape(b,h*w).sum(-1)>=self.match_sample_num)[:,None], 
                              choose1, choose_backup1)
        choose1_match = torch.gather(match1, dim=-1, index=choose1)
        
        choose2 = torch.topk(-coord_diff2, k = 128, dim = -1)[1]
        choose2 = torch.where((mask2.reshape(b,h*w).sum(-1)>=self.match_sample_num)[:,None], 
                              choose2, choose_backup2)
        choose2_match = torch.gather(match2, dim=-1, index=choose2)
        
        

        assert (torch.gather(mask2.reshape(b,h*w), dim = -1, index = choose1_match)>0).all()
        assert (torch.gather(mask2.reshape(b,h*w), dim = -1, index = choose2)>0).all()
        assert (torch.gather(mask1.reshape(b,h*w), dim = -1, index = choose2_match)>0).all()
        assert (torch.gather(mask1.reshape(b,h*w), dim = -1, index = choose1)>0).all()

        match1 = torch.stack([choose1, choose1_match],dim = -1)
        match2 = torch.stack([choose2, choose2_match],dim = -1)




        
        
        
        return match1, match2 #b x s

    def choose_from_feature(self, feature, choose,h,w):

        
        # ts = time()

        coordinate_orig = self.choose_to_coordinate(choose, h,w)
        # print(' choose_roi_to_coordinate', time()-ts)
        # ts = time()
        coordinate = coordinate_orig//self.extractor_scale
        b, c, h, w = feature.shape
        # print(' rescale coordinate', time()-ts)
        # ts = time()
        choose = self.coordinate_to_choose(coordinate, h, w).contiguous()
        # print(' coordinate_to_choose', time()-ts)
        # ts = time()
        
        
        # mask = torch.zeros_like(feature)
        # mask.scatter_(2, choose, 1.)
        # print(' create mask', time()-ts)
        # ts = time()

        feature = feature.permute(0,2,3,1).view(b,h*w, c).contiguous()
        # print(' feature.view', time()-ts)
        # ts = time()

        choose = choose[..., None].expand(-1, -1, feature.size(2))    ## expanding index
        ret = torch.gather(feature, dim=1, index=choose).squeeze()
        #ret = feature[torch.arange(b)[:,None].contiguous(), :, choose] #b x c x sample
        #ret = feature[mask]
        # print(' choose from feature', time()-ts)
        # ts = time()

        return ret
    

    def scale_up_choose(self, choose, h, w, scale_factor):
        return self.coordinate_to_choose(
            self.choose_to_coordinate(choose, h, w)*scale_factor, 
            w *scale_factor,
            h * scale_factor)
    def scale_down_choose(self, choose, h, w, scale_factor):
        return self.coordinate_to_choose(
            self.choose_to_coordinate(choose, h, w)//scale_factor, 
            w //scale_factor,
            h //scale_factor)


    def choose_to_coordinate(self, match_choose, h, w):
        r = match_choose//w
        c = match_choose%w
        return torch.stack([r,c], dim = -1)

        
    def forward(self, inputs):
        
        
        
        rgb_raw1, rgb_raw2 = inputs['rgb_raw'][:,0], inputs['rgb_raw'][:,1]
        
        rgb_raw = torch.concatenate([rgb_raw1, rgb_raw2], dim = 0).contiguous()
        pts_raw1, pts_raw2 = inputs['pts_raw'][:,0], inputs['pts_raw'][:,1]

        choose_backup1, choose_backup2 = inputs['choose'][:,0], inputs['choose'][:,1]
        
       
        mask1 ,mask2 = inputs['mask'][:,0], inputs['mask'][:,1]
       
        
        b,_,_,_ = pts_raw1.shape
        
        rgb1, rgb2 = inputs['rgb'][:,0], inputs['rgb'][:,1]
        pts1, pts2 = inputs['pts'][:,0], inputs['pts'][:,1]
        rotation_ref = inputs['rotation_ref']

        pts2 = self.rotate_pts_batch(pts2, rotation_ref.transpose(1,2))
       
        
        
        #print('extract_feature')
        #dino_feature = self.extract_feature(rgb_raw) 
        
        

        
       
        
        #dino_feature1, dino_feature2 = dino_feature[:b], dino_feature[b:]
       


        
        #print('matching')
        # matchpair1, matchpair2 = self.matcher(dino_feature1.permute(0,2,3,1), 
        #                             dino_feature2.permute(0,2,3,1), 
        #                             mask1[:,:,:,None], 
        #                             mask2[:,:,:,None],
        #                             choose_backup1, choose_backup2)
        
        
        # self1, match1 = matchpair1[:,:,0], matchpair1[:,:,1]
        # self2, match2 = matchpair2[:,:,0], matchpair2[:,:,1]

        

        
        
        
        
        # pts_raw1 = pts_raw1.reshape(b,self.num_patches * self.num_patches,3)
        # pts_raw2 = pts_raw2.reshape(b,self.num_patches * self.num_patches,3)
        # rgb_raw1 = rgb_raw1.reshape(b,self.num_patches * self.num_patches,3)
        # rgb_raw2 = rgb_raw2.reshape(b,self.num_patches * self.num_patches,3)


        # self_pts1 = pts_raw1[torch.arange(b)[:,None], self1,:]
        # self_rgb1 = rgb_raw1[torch.arange(b)[:,None], self1,:]
        # match_pts1 = pts_raw2[torch.arange(b)[:,None], match1,:]

        # self_pts2 = pts_raw2[torch.arange(b)[:,None], self2,:]
        # self_rgb2 = rgb_raw1[torch.arange(b)[:,None], self2,:]
        # match_pts2 = pts_raw1[torch.arange(b)[:,None], match2,:]

        

        # import pdb;pdb.set_trace()








        # match_pts1 = torch.concatenate([pts1, match_pts1], dim = -2)
        # match_pts2 = torch.concatenate([pts2, match_pts2], dim = -2)

        # pts1 = torch.concatenate([pts1, self_pts1], dim = -2)
        # pts2 = torch.concatenate([pts2, self_pts2], dim = -2)

        # import pdb;pdb.set_trace()

        # drift1 =  match_pts1 - pts1
        # drift2 =  match_pts2 - pts2
        
        
        # rgb1 = torch.concatenate([rgb1, self_rgb1], dim = -2)
        # rgb2 = torch.concatenate([rgb2, self_rgb2], dim = -2)
        
        

        # dis_map1, rgb_map1 = self.feat2smap(pts1, torch.concatenate([rgb1, drift1], dim = -1))
        # dis_map2, rgb_map2 = self.feat2smap(pts2, torch.concatenate([rgb2, drift2], dim = -1))
        dis_map1, rgb_map1 = self.feat2smap(pts1, rgb1)
        dis_map2, rgb_map2 = self.feat2smap(pts2, rgb2)
       
        
        # backbone
        x1 = self.spherical_fpn(dis_map1, rgb_map1)
        x2 = self.spherical_fpn(dis_map2, rgb_map2)
        
        x = torch.cat([x1, x2], axis = 1)
        # viewpoint rotation
        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)
        pred_vp_rot = self.v_branch._get_vp_rotation(rho_prob, phi_prob,{})

        # in-plane rotation
        # drift1 =  match_pts1 - (vp_rot[:,None,:,:] @ pts1[:,:,:,None]).squeeze()
        
        # drift2 =  (vp_rot[:,None,:,:] @ match_pts2[:,:,:,None]).squeeze() - pts2
        
        
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
