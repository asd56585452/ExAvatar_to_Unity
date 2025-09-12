import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.renderer import TexturesUV
from config import cfg

class MyGroupNorm(nn.Module):
    """
    一個自訂的 GroupNorm 層，專門處理 2D 輸入以相容 ONNX 導出。
    它在內部將 2D 輸入 (N, C) 暫時轉為 4D (N, C, 1, 1)，
    執行 GroupNorm 後再轉回 2D。
    對於 3D 或更高維度的輸入，它的行為和標準 nn.GroupNorm 完全一樣。
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(MyGroupNorm, self).__init__()
        # 建立一個標準的 GroupNorm 層實例
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x):
        # 檢查輸入張量的維度
        if x.dim() == 2:
            # 如果是 2D 輸入 [N, C]
            # 1. 增加維度 -> [N, C, 1, 1]
            reshaped_x = x.unsqueeze(-1).unsqueeze(-1)
            # 2. 執行標準的 GroupNorm
            normed_x = self.gn(reshaped_x)
            # 3. 壓平維度 -> [N, C]
            return normed_x.squeeze(-1).squeeze(-1)
        else:
            # 如果是 3D, 4D, 5D... 輸入，直接執行
            return self.gn(x)

    # 讓這個模組的 state_dict 和內部的 gn 層保持一致
    # 這一步驟是可選的，但能讓結構更清晰
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 將權重直接載入到 self.gn 中
        self.gn._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def state_dict(self, *args, **kwargs):
        # 回傳 self.gn 的 state_dict
        return self.gn.state_dict(*args, **kwargs)

def make_linear_layers(feat_dims, relu_final=True, use_gn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_gn:
                layers.append(MyGroupNorm(4, feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def get_face_index_map_xy(mesh, face, cam_param, render_shape):
    batch_size = mesh.shape[0]
    face = torch.from_numpy(face).cuda()[None,:,:].repeat(batch_size,1,1)
    mesh = torch.stack((-mesh[:,:,0], -mesh[:,:,1], mesh[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(mesh, face)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                principal_point=cam_param['princpt'],
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    outputs = rasterizer(mesh)
    return outputs

class MeshRenderer(nn.Module):
    def __init__(self, vertex_uv, face_uv):
        super(MeshRenderer, self).__init__()
        self.vertex_uv = torch.FloatTensor(vertex_uv).cuda()
        self.face_uv = torch.LongTensor(face_uv).cuda()

    def forward(self, uvmap, mesh, face, cam_param, render_shape):
        batch_size, uvmap_dim, uvmap_height, uvmap_width = uvmap.shape
        render_height, render_width = render_shape

        # get visible faces from mesh
        mesh = torch.bmm(cam_param['R'], mesh.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3) # world coordinate -> camera coordinate
        fragments = get_face_index_map_xy(mesh, face, cam_param, (render_height, render_width))
        vertex_uv = torch.stack((self.vertex_uv[:,0], 1 - self.vertex_uv[:,1]),1)[None,:,:].repeat(batch_size,1,1) # flip y-axis following PyTorch3D convention
        renderer = TexturesUV(uvmap.permute(0,2,3,1), self.face_uv[None,:,:].repeat(batch_size,1,1), vertex_uv)
        render = renderer.sample_textures(fragments) # batch_size, render_height, render_width, faces_per_pixel, uvmap_dim
        render = render[:,:,:,0,:].permute(0,3,1,2) # batch_size, uvmap_dim, render_height, render_width
        
        # fg mask
        pix_to_face = fragments.pix_to_face # batch_size, render_height, render_width, faces_per_pixel. invalid: -1
        pix_to_face_xy = pix_to_face[:,:,:,0] # Note: this is a packed representation

        # packed -> unpacked
        is_valid = (pix_to_face_xy != -1).float()
        pix_to_face_xy = (pix_to_face_xy - torch.arange(batch_size)[:,None,None].cuda() * face.shape[0]) * is_valid + (-1) * (1 - is_valid)
        pix_to_face_xy = pix_to_face_xy.long()
        
        # make backgroud pixels to -1
        render[pix_to_face_xy[:,None,:,:].repeat(1,uvmap_dim,1,1) == -1] = -1
        return render


