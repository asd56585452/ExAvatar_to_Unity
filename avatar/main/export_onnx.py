import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
#python export_onnx.py --subject_id gyeongsik --test_epoch 4 --motion_path /home/cgvmis418/ExAvatar_to_Unity/motions/jungkook_standing_next_to_you --output_path "../data/NeuMan/data/gyeongsik/human_model_fix.onnx"
#python export_onnx.py --subject_id gyeongsik --test_epoch 4 --motion_path /home/cgvmis418/ExAvatar_to_Unity/motions/jungkook_standing_next_to_you --output_path "../data/NeuMan/data/gyeongsik/human_model_ChunkedGroupNorm.onnx"
# --- 為 aten::sinc 定義翻譯規則 (修正版) ---
@parse_args("v")
def symbolic_sinc(g, x):
    """
    Symbolic function for aten::sinc.
    This function defines how to translate torch.sinc(x) into basic ONNX operators.
    This version is robust to different dtypes (float, half, double).
    """
    # 1. 建立標準的 float32 常數
    zero_const_float = g.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32))
    one_const_float = g.op("Constant", value_t=torch.tensor(1.0, dtype=torch.float32))

    # 2. 使用 CastLike 將常數轉換為與輸入 x 相同的資料類型
    zero_const = g.op("CastLike", zero_const_float, x)
    one_const = g.op("CastLike", one_const_float, x)
    
    # 3. 執行原始的數學邏輯
    sin_x = g.op("Sin", x)
    div_result = g.op("Div", sin_x, x)
    is_zero = g.op("Equal", x, zero_const)
    
    # 4. 使用 Where 運算子處理 x=0 的情況
    return g.op("Where", is_zero, one_const, div_result)

# 將我們的翻譯規則註冊給 ONNX 匯出器
register_custom_op_symbolic("aten::sinc", symbolic_sinc, 9)
import torch
import json
import argparse
import os.path as osp

# 假設您的 config, base, model, smpl_x 模組都在可導入的路徑中
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from model import get_model # 您需要確保 model.py 和相關依賴項可以被正確匯入

# --- 步驟 1: 建立一個包裝模型 (Wrapper Model) 來處理字典輸入 ---
# torch.onnx.export 不直接支援字典輸入，我們需要這個包裝器來將張量列表轉換回字典
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        with torch.no_grad():
            mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = model.module.human_gaussian.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
            joint_zero_pose = model.module.human_gaussian.get_zero_pose_human()

            # extract triplane feature
            tri_feat = model.module.human_gaussian.extract_tri_feature()
        
            # get Gaussian assets
            geo_feat = model.module.human_gaussian.geo_net(tri_feat)
            mean_offset = model.module.human_gaussian.mean_offset_net(geo_feat) # mean offset of Gaussians
            scale = model.module.human_gaussian.scale_net(geo_feat) # scale of Gaussians
            rgb = model.module.human_gaussian.rgb_net(tri_feat) # rgb of Gaussians
            mean_3d = mesh_neutral_pose + mean_offset # 大 pose
        # --- 核心修改：將傳入的常數張量註冊為 buffer ---
        self.register_buffer('tri_feat', tri_feat)
        self.register_buffer('scale', scale)
        self.register_buffer('rgb', rgb)
        self.register_buffer('mean_3d', mean_3d)
        self.register_buffer('mesh_neutral_pose_wo_upsample', mesh_neutral_pose_wo_upsample)
        self.register_buffer('transform_mat_neutral_pose', transform_mat_neutral_pose)
        
        # 根據 smplx_params_smoothed_0.json 和 cam_params_0.json 的結構，定義輸入張量的鍵名和順序
        # **這個順序必須與後面建立 dummy_inputs 的順序完全一致**
        self.smplx_keys = [
            'root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 
            'lhand_pose', 'rhand_pose', 'expr', 'trans'
        ]
        # 注意: cam_params_0.json 中的 't' 在 module.py 中被當作 cam_param['t'] 使用，
        # 但 smplx_params_smoothed_0.json 中也有 'trans'。為避免混淆，請確認您的模型確實如此使用。
        # 根據 cam_params_0.json 的內容，這裡的鍵應為 'R', 't', 'focal', 'princpt'。
        self.cam_keys = ['R', 't', 'focal', 'princpt']

    def forward(self, *inputs):
        # 將傳入的扁平化張量元組 (tuple of tensors) 重新組合成字典
        smplx_param = {}
        cam_param = {}
        
        smplx_input_count = len(self.smplx_keys)
        cam_inputs_count = len(self.cam_keys)
        smplx_inputs_tuple = inputs[:smplx_input_count]
        cam_inputs_tuple = inputs[smplx_input_count:smplx_input_count+cam_inputs_count]
        # joint_zero_pose = self.model.module.human_gaussian.get_zero_pose_human()

        for i, key in enumerate(self.smplx_keys):
            smplx_param[key] = smplx_inputs_tuple[i]
            
        for i, key in enumerate(self.cam_keys):
            cam_param[key] = cam_inputs_tuple[i]

        # 呼叫原始模型的 human_gaussian 部分
 
        # get pose-dependent Gaussian assets
        mean_offset_offset, scale_offset = self.model.module.human_gaussian.forward_geo_network(self.tri_feat, smplx_param)
        scale, scale_refined = torch.exp(self.scale).repeat(1,3), torch.exp(self.scale+scale_offset).repeat(1,3)
        mean_combined_offset, mean_offset_offset = self.model.module.human_gaussian.get_mean_offset_offset(smplx_param, mean_offset_offset)
        mean_3d_refined = self.mean_3d + mean_combined_offset # 大 pose

        # smplx facial expression offset
        smplx_expr_offset = (smplx_param['expr'][None,None,:] * self.model.module.human_gaussian.expr_dirs).sum(2)
        mean_3d = self.mean_3d + smplx_expr_offset # 大 pose
        mean_3d_refined = mean_3d_refined + smplx_expr_offset # 大 pose

        # get nearest vertex
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        # nn_vertex_idxs = knn_points(mean_3d[None,:,:], self.mesh_neutral_pose_wo_upsample[None,:,:], K=1, return_nn=True).idx[0,:,0] # dimension: smpl_x.vertex_num_upsampled
        # nn_vertex_idxs = self.model.module.human_gaussian.lr_idx_to_hr_idx(nn_vertex_idxs)
        # mask = (self.model.module.human_gaussian.is_rhand + self.model.module.human_gaussian.is_lhand + self.model.module.human_gaussian.is_face) > 0
        # updates = torch.arange(smpl_x.vertex_num_upsampled, device=nn_vertex_idxs.device, dtype=torch.int64)
        # nn_vertex_idxs = torch.where(mask, updates, nn_vertex_idxs)

        # get transformation matrix of the nearest vertex and perform lbs
        # transform_mat_joint = self.model.module.human_gaussian.get_transform_mat_joint(self.transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        # transform_mat_vertex = self.model.module.human_gaussian.get_transform_mat_vertex(transform_mat_joint, nn_vertex_idxs)
        # mean_3d = self.model.module.human_gaussian.lbs(mean_3d, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param
        # mean_3d_refined = self.model.module.human_gaussian.lbs(mean_3d_refined, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param
        
        # forward to rgb network
        rgb = (torch.tanh(self.rgb) + 1) / 2
        
        rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smpl_x.vertex_num_upsampled,1,1)) # constant rotation
        opacity = torch.ones((smpl_x.vertex_num_upsampled,1)).float().cuda() # constant opacity
        # 根據 module.py 的定義，human_asset 是一個字典。
        # ONNX 導出需要返回一個張量或張量的元組，因此我們提取字典中的所有張量。
        return (
            mean_3d,
            opacity,
            scale,
            rotation, 
            rgb,
            mean_3d_refined,
            scale_refined,
            self.mesh_neutral_pose_wo_upsample,
            self.transform_mat_neutral_pose
        )

def main():
    # --- 與 animate.py 類似的參數設定 ---
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument('--subject_id', type=str, required=True, help="Subject ID for configuration")
    parser.add_argument('--test_epoch', type=str, required=True, help="Epoch number of the model checkpoint to load")
    parser.add_argument('--motion_path', type=str, required=True, help="Path to the motion data containing smplx and camera parameters")
    parser.add_argument('--output_path', type=str, default='human_gaussian_model.onnx', help="Path to save the output ONNX model")
    args = parser.parse_args()

    cfg.set_args(args.subject_id)

    # --- 步驟 2: 載入原始 PyTorch 模型 ---
    print("正在載入 PyTorch 模型...")
    tester = Tester(args.test_epoch)
    
    # 載入 animate.py 中使用的 ID 資訊
    root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id)
    with open(osp.join(root_path, 'smplx_optimized', 'shape_param.json')) as f:
        shape_param = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'face_offset.json')) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'joint_offset.json')) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'locator_offset.json')) as f:
        locator_offset = torch.FloatTensor(json.load(f))
    smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)
    
    tester.smplx_params = None  # 設為 None 以從 base.py 中的 ckpt 載入
    tester._make_model()
    tester.model.eval()  # 確保模型處於評估模式
    print(f"模型 snapshot_{args.test_epoch}.pth 載入完成。")

    # --- 步驟 3: 準備範例輸入 (Dummy Input) ---
    print("正在準備範例輸入...")
    # 使用 motion_path 中的第一偵數據作為範例輸入
    frame_idx = 0
    cam_param_file = osp.join(args.motion_path, 'cam_params', f'{frame_idx}.json')
    smplx_param_file = osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', f'{frame_idx}.json')

    with open(cam_param_file) as f:
        cam_param_dict = {k: torch.FloatTensor(v).cuda() for k, v in json.load(f).items()}
    with open(smplx_param_file) as f:
        smplx_param_dict = {k: torch.FloatTensor(v).cuda().view(-1) for k, v in json.load(f).items()}

    # 建立包裝模型
    wrapped_model = ModelWrapper(tester.model).cuda().eval()

    # 按照 ModelWrapper 中定義的順序，將字典轉換為張量元組
    smplx_inputs_tuple = tuple(smplx_param_dict[key] for key in wrapped_model.smplx_keys)
    cam_inputs_tuple = tuple(cam_param_dict[key] for key in wrapped_model.cam_keys)
    dummy_inputs = smplx_inputs_tuple + cam_inputs_tuple 
    # get_neutral_pose_human_input = tester.model.module.human_gaussian.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
    # dummy_inputs = smplx_inputs_tuple + cam_inputs_tuple + get_neutral_pose_human_input
    
    # 定義輸入和輸出的名稱 (這在之後使用 ONNX 模型時很重要)
    input_names = wrapped_model.smplx_keys + wrapped_model.cam_keys
    output_names = [
        'mean_3d',
            'opacity',
            'scale',
            'rotation', 
            'rgb',
            'mean_3d_refined',
            'scale_refined',
            'mesh_neutral_pose_wo_upsample',
            'transform_mat_neutral_pose'

    ] # 根據您在 Wrapper 中返回的內容命名
    print("範例輸入準備完成。")

    # --- 步驟 4: 執行 ONNX 轉換 ---
    print(f"開始將模型轉換為 ONNX 格式，並儲存至 {args.output_path}...")
    torch.onnx.export(
        wrapped_model,
        dummy_inputs,
        args.output_path,
        input_names=input_names,
        output_names=output_names,
        verbose=False, # 設為 True 可以看到詳細的轉換日誌
        opset_version=16, # 建議使用 11 或更高的版本
        export_params=True
    )
    print("模型轉換成功！")

if __name__ == "__main__":
    main()