import torch
import torch.nn as nn
import torch.nn.functional as F

from module import SphericalFPN, V_Branch, I_Branch, I_Branch_Pair
from extractor_dino import ViTExtractor
from loss import SigmoidFocalLoss
from module import PointNet2MSG
from smap_utils import Feat2Smap
from torchvision import transforms
from rotation_utils import angle_of_rotation, Ortho6d2Mat
def plot_pt(pt,index):
    import pyvista as pv
    import matplotlib.pyplot as plt
    cloud = pv.PolyData(pt[index].cpu().numpy())
    mesh = cloud.delaunay_2d()
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white')
    _ = plotter.add_axes(box=True)
    
    
    plotter.show()

def plot_pt_pair(pt1, pt2,self, match,index, m_index):
    import pyvista as pv
    import matplotlib.pyplot as plt
    cloud1 = pv.PolyData(pt1[index].cpu().numpy())
    mesh1 = cloud1.delaunay_2d()
    cloud2 = pv.PolyData(pt2[index].cpu().numpy() +0.5)
    mesh2 = cloud2.delaunay_2d()

    m_cloud1 = pv.PolyData(self[index][[m_index]].cpu().numpy())
    # m_mesh1 = m_cloud1.delaunay_2d()
    m_cloud2 = pv.PolyData(match[index][[m_index]].cpu().numpy() +0.5)
    # m_mesh2 = m_cloud2.delaunay_2d()

    plotter = pv.Plotter()
    plotter.add_mesh(mesh1, color='white')
    plotter.add_mesh(mesh2, color='blue')
    plotter.add_points(m_cloud1, color='red')
    plotter.add_points(m_cloud2, color='red')


    _ = plotter.add_axes(box=True)
    
    
    plotter.show()

def plot_rgb(rgb, index):
    from matplotlib import pyplot as plt
    plt.imshow(rgb[index].reshape(210,210,-1).cpu().numpy(), interpolation='nearest')
    plt.show()

class Net(nn.Module):
    def __init__(self, resolution=64, ds_rate=2, num_patches = 15):
        super(Net, self).__init__()
        self.res = resolution
        self.ds_rate = ds_rate
        self.ds_res = resolution//ds_rate
        extractor = ViTExtractor('dinov2_vits14', 14, device = 'cuda')
        self.extractor =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        
        # self.extractor.model =  torch.nn.DataParallel(self.extractor.model, range(2))
        self.extractor_preprocess = transforms.Normalize(mean=extractor.mean, std=extractor.std)
        self.extractor_layer = 11
        self.extractor_facet = 'token'
        self.pn2msg = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]], dim_in = 384)
        
        self.num_patches = num_patches


        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.soft_max = nn.Softmax(dim=-1)

        # data processing
        self.feat2smap = Feat2Smap(self.res)
        self.feat2smap_drift = Feat2Smap(self.res//self.ds_rate)
        # self.spherical_fpn_drift = SphericalFPN(ds_rate=self.ds_rate, dim_in1=1, dim_in2=3)
        self.spherical_fpn = SphericalFPN(ds_rate=self.ds_rate, dim_in1=1, dim_in2=3)
        self.v_branch = V_Branch(resolution=self.ds_res, in_dim = 256)
        self.i_branch = I_Branch(resolution=self.ds_res, in_dim = 256*2)
        self.match_threshould = nn.Parameter(torch.tensor(-1.0, requires_grad=True))
        self.mlp  = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.rotation_estimator = nn.Sequential(
           
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

        self.feature_mlp = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
        )

        self.similarity_mlp = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1),
        )
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
            dino_feature = self.extractor.forward_features(rgb_raw)["x_prenorm"][:,1:]
            # dino_feature = self.extractor.extract_descriptors(rgb_raw, layer = self.extractor_layer, facet = self.extractor_facet )
        
        dino_feature = dino_feature.reshape(dino_feature.shape[0],self.num_patches,self.num_patches,-1)
        
        return dino_feature.contiguous() # b x c x h x w
    
    
    def matcher_cyclic_pair(self, feature1, feature2, mask1, mask2, choose_backup1, choose_backup2):
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
    
    def similarity_weights(self, feature1, feature2):
        #feature: b x h x w x c
        #mask: b x h x w x 1
        
        
        
        b,s,c= feature2.shape
        feature1 = feature1 / feature1.norm(dim=-1, keepdim=True)
        feature2 = feature2 / feature2.norm(dim=-1, keepdim=True)
        feature2 = torch.cat([feature2, 
                              torch.zeros_like(feature2[0,0])[None,None,:].repeat((feature2.shape[0],1,1))], dim = 1)
        
        with torch.no_grad():
            cos_sim = self.cos(feature1[:,:,None,:], feature2[:,None,:,:])
            weights = self.soft_max(cos_sim)
        # cos_sim = 2 - torch.cdist(feature1, feature2)
        # similarity_feature = torch.cat((feature1[:,:,None,:].repeat(1,1,feature2.shape[1],1), 
        #                                          feature2[:,None,:,:].repeat(1,feature1.shape[1],1,1)), dim = -1)
        
        # samples = similarity_feature.shape[1]
        # similarity_feature = similarity_feature.reshape(similarity_feature.shape[0],-1, similarity_feature.shape[-1])
        # similarity_feature = similarity_feature.transpose(1,2)
        # weights = self.similarity_mlp(similarity_feature).transpose(1,2)
        # weights = weights.reshape(similarity_feature.shape[0],samples,samples)


        # match = torch.argmin(cos_sim, dim = -1) # b x n x n
                
        return weights#, match2 #b x s
    def match_enforcer(self, match_feature):
        r = self.rotation_estimator(match_feature)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1,3,3)
        return r

    def best_match(self, feature1, feature2):
        #feature: b x h x w x c
        #mask: b x h x w x 1
        
        
        
        b,s,c= feature2.shape
        feature1 = feature1 / feature1.norm(dim=-1, keepdim=True)
        feature2 = feature2 / feature2.norm(dim=-1, keepdim=True)
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
        with torch.no_grad():
            cos_sim = cos(feature1[:,:,None,:], feature2[:,None,:,:])
            
        
        

        cos_sim_m, match = torch.max(cos_sim, dim = -1) # b x n x n
                
        return match,cos_sim_m, cos_sim #, match2 #b x s


    def forward(self, inputs):
        # import pdb;pdb.set_trace()
        rgb1, rgb2 = inputs['rgb'][:,0,:,:], inputs['rgb'][:,1,:,:]
        pts1, pts2 = inputs['pts'][:,0,:,:], inputs['pts'][:,1,:,:]
        # mask1, mask2 = inputs['mask'][:,0], inputs['mask'][:,1]
        # choose1, choose2 = inputs['choose'][:,0], inputs['choose'][:,1]
        b,_,rgb_h,rgb_w,_ = inputs['rgb_raw'].shape
        rotation_ref = inputs['rotation_ref']

        rgb_raw = inputs['rgb_raw'].reshape(-1,14*self.num_patches,14*self.num_patches,3)
        # rgb_raw1, rgb_raw2 = rgb_raw.reshape(b,2,rgb_h, rgb_w)
       
        pts2 = self.rotate_pts_batch(pts2, rotation_ref.transpose(1,2))
        #import pdb;pdb.set_trace()
        feature = self.extract_feature(rgb_raw).reshape(b*2,(self.num_patches)**2,-1)
        _,_,num_sample = inputs['choose'].shape
        match_num = 100
        choose = inputs['choose'][:,:,:match_num]
        feature = feature[torch.arange(b*2)[:,None], 
                          choose.reshape(b*2,match_num),:].reshape(b,2,match_num,-1)
        
        feature1, feature2 = feature[:,0], feature[:,1]
        pts_raw = inputs['pts_raw']
        pts_raw = pts_raw.reshape(b*2,(self.num_patches)**2,-1)[torch.arange(b*2)[:,None], choose.reshape(b*2,match_num),:].reshape(b,2,match_num,-1)
        ptsf1, ptsf2 = pts_raw[:,0], pts_raw[:,1]
        ptsf2 = self.rotate_pts_batch(ptsf2, rotation_ref.transpose(1,2))

        # feature2 = feature1
        # ptsf2 = self.rotate_pts_batch(ptsf1, inputs['rotation_label'].transpose(1,2))


        # pnfeature1 = self.pn2msg(torch.cat([ptsf1, feature1], dim=2)).transpose(1,2)
        # pnfeature1_global = torch.mean(pnfeature1, 1, keepdim=True)
        # pnfeature1 = torch.cat([pnfeature1, pnfeature1_global.repeat(1,pnfeature1.shape[1],1)], dim = 2)
        
        # pnfeature2 = self.pn2msg(torch.cat([ptsf2, feature2], dim=2)).transpose(1,2)
        # pnfeature2_global = torch.mean(pnfeature2, 1, keepdim=True)
        # pnfeature2 = torch.cat([pnfeature2, pnfeature2_global.repeat(1,pnfeature2.shape[1],1)], dim = 2)


        # sim_weights = self.similarity_weights(pnfeature1, pnfeature2)
        # match =sim_weights@pnfeature2
        
        # match_feature = self.mlp(  torch.cat([pnfeature1, match], dim=2).transpose(1,2))
        # r1 = self.rotation_estimator(match_feature.squeeze())
        # r1 = Ortho6d2Mat(r1[:,:3], r1[:,3:])
        # match_feature = torch.zeros_like(match_feature)
        # import pdb;pdb.set_trace()

        # pnfeature1 = self.pn2msg(torch.cat([ptsf1, feature1], dim=2)).transpose(1,2)
        # pnfeature2 = self.pn2msg(torch.cat([ptsf2, feature2], dim=2)).transpose(1,2)
        # best_match, cos_sim_m, cos_sim = self.best_match(feature1, feature2)

        # match = ptsf2[torch.arange(b)[:,None], best_match,:]
        # match = torch.where(cos_sim_m[:,:,None]>self.match_threshould, match, 0)
        # import pdb;pdb.set_trace()
        # match_feature = self.mlp(  torch.cat([pnfeature1, match], dim=2).transpose(1,2))
        # r1 = self.rotation_estimator(match_feature)
        # match_feature = match_feature.transpose(1,2)
        
        
        # try:

        #     print(angle_of_rotation(inputs['rotation_ref'] @ inputs['rotation_label'].transpose(1,2)).mean())
        # except:
        #     pass
        # pts1_plot = self.rotate_pts_batch(pts1, inputs['rotation_label'].transpose(1,2))
        # ptsf1_plot = self.rotate_pts_batch(ptsf1, inputs['rotation_label'].transpose(1,2))
        # with torch.no_grad():
        #     print((torch.arccos(self.cos(ptsf1_plot, match))*180/3.1415926).mean())
        
        # import pdb;pdb.set_trace()
        # plot_pt_pair(pts1_plot, pts2, ptsf1_plot, match, 0,0)

        
        
        dis_map, rgb_map= self.feat2smap(pts1, rgb1)
        # dis_map2, match_map = self.feat2smap(ptsf1, match)
        # import pdb;pdb.set_trace()
        # match_map = torch.zeros_like(match_map)
        # dis_map1 = torch.zeros_like(dis_map1)
        
        # import pdb;pdb.set_trace()
        # backbone
        x = self.spherical_fpn(dis_map, rgb_map)
        # x1 = torch.cat([x, torch.tile(match_feature[:,:,:,None],(1,1,x.shape[2], x.shape[3]))], dim = 1)
        

        # x2 = self.spherical_fpn(dis_map2, rgb_map2)
        
        # x = torch.cat([x1, x2], axis = 1)
        # viewpoint rotation
        vp_rot, rho_prob, phi_prob = self.v_branch(x, inputs)
        pred_vp_rot = self.v_branch._get_vp_rotation(rho_prob, phi_prob,{})

        
        # in-plane rotation
        ptsf1  = self.rotate_pts_batch(ptsf1, vp_rot.transpose(1,2))
        pnfeature1 = self.pn2msg(torch.cat([ptsf1, feature1], dim=2)).transpose(1,2)
        pnfeature1_global = torch.mean(pnfeature1, 1, keepdim=True)
        pnfeature1 = torch.cat([pnfeature1, pnfeature1_global.repeat(1,pnfeature1.shape[1],1)], dim = 2)
        pnfeature2 = self.pn2msg(torch.cat([ptsf2, feature2], dim=2)).transpose(1,2)
        pnfeature2_global = torch.mean(pnfeature2, 1, keepdim=True)
        pnfeature2 = torch.cat([pnfeature2, pnfeature2_global.repeat(1,pnfeature2.shape[1],1)], dim = 2)
        
        pnfeature1 = self.feature_mlp(pnfeature1.transpose(1,2)).transpose(1,2)
        pnfeature2 = self.feature_mlp(pnfeature2.transpose(1,2)).transpose(1,2)

        sim_weights = self.similarity_weights(pnfeature1, pnfeature2)
        pnfeature2 = torch.cat([pnfeature2, 
                              torch.zeros_like(pnfeature2[0,0])[None,None,:].repeat((pnfeature2.shape[0],1,1))], dim = 1)
        
        match = sim_weights@pnfeature2
        
        # import pdb;pdb.set_trace()
        match_feature = self.mlp(  torch.cat([pnfeature1, match], dim=2).transpose(1,2))
        r2 = self.rotation_estimator(match_feature.squeeze())
        r2 = Ortho6d2Mat(r2[:,:3], r2[:,3:])
        # match_feature = torch.zeros_like(match_feature)
        # match_feature = match_feature.transpose(1,2)

        x2 = torch.cat([x, torch.tile(match_feature[:,:,:,None],(1,1,x.shape[2], x.shape[3]))], dim = 1)

        ip_rot = self.i_branch(x2, vp_rot)

        outputs = {
            'pred_rotation': vp_rot @ ip_rot,
            'pred_vp_rotation': pred_vp_rot,
            'pred_ip_rotation': ip_rot,
            'rho_prob': rho_prob,
            'phi_prob': phi_prob,
            'thres': self.match_threshould,
            # 'r1': r1,
            'r2':vp_rot @ r2
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

        # r1_loss = self.l1loss(pred['r1'], gt['rotation_label'])
        r2_loss = self.l1loss(pred['r2'], gt['rotation_label'])
        residual_angle = angle_of_rotation(pred['pred_rotation'].transpose(1,2) @ gt['rotation_label'])
        # r1_residual_angle = angle_of_rotation(pred['r1'].transpose(1,2) @ gt['rotation_label'])
        r2_residual_angle = angle_of_rotation(pred['r2'].transpose(1,2) @ gt['rotation_label'])

        vp_residual_angle = angle_of_rotation(pred['pred_vp_rotation'].transpose(1,2) @ gt['vp_rotation_label'])
        ip_residual_angle = angle_of_rotation(pred['pred_ip_rotation'].transpose(1,2) @ gt['ip_rotation_label'])

        loss = self.cfg.vp_weight * vp_loss + ip_loss  + r2_loss

        return {
            'loss': loss,
            'vp_loss': vp_loss,
            'ip_loss': ip_loss,
            'r2_loss': r2_loss,
            'rho_acc': rho_acc,
            'phi_acc': phi_acc,
            'residual_angle':residual_angle.mean(),
            # 'r1_residual_angle':r1_residual_angle.mean(),
            'r2_residual_angle':r2_residual_angle.mean(),
            'vp_residual_angle':vp_residual_angle.mean(),
            'ip_residual_angle': ip_residual_angle.mean(),
            'match_thres': pred['thres'].mean()
        }
