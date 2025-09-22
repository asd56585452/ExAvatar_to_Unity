import torch
import json
import argparse
import os.path as osp
import numpy as np
import onnxruntime # 載入 ONNX 執行環境

# 新增必要的 import，與 export_onnx.py 同步
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.ops import knn_points

# 假設您的 config, base, model, smpl_x 模組都在可導入的路徑中
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from model import get_model

# --- 步驟 1: 使用與 export_onnx.py 完全相同的 ModelWrapper ---
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
             'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 
            'lhand_pose', 'rhand_pose', 'expr'
        ]
        # 注意: cam_params_0.json 中的 't' 在 module.py 中被當作 cam_param['t'] 使用，
        # 但 smplx_params_smoothed_0.json 中也有 'trans'。為避免混淆，請確認您的模型確實如此使用。
        # 根據 cam_params_0.json 的內容，這裡的鍵應為 'R', 't', 'focal', 'princpt'。
        self.cam_keys = []

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
    parser = argparse.ArgumentParser(description="Verify ONNX model against PyTorch model")
    parser.add_argument('--subject_id', type=str, required=True, help="Subject ID")
    parser.add_argument('--test_epoch', type=str, required=True, help="Model checkpoint epoch")
    parser.add_argument('--motion_path', type=str, required=True, help="Path to motion data")
    parser.add_argument('--onnx_path', type=str, default='human_gaussian_model.onnx', help="Path to the ONNX model to verify")
    args = parser.parse_args()

    cfg.set_args(args.subject_id)

    print("正在載入 PyTorch 模型並準備範例輸入...")
    tester = Tester(args.test_epoch)
    
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
    
    tester.smplx_params = None
    tester._make_model()
    tester.model.eval()

    # *** 建議：使用與 export_onnx.py 相同的 frame_idx 以確保輸入完全一致 ***
    frame_idx = 100
    cam_param_file = osp.join(args.motion_path, 'cam_params', f'{frame_idx}.json')
    smplx_param_file = osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', f'{frame_idx}.json')

    with open(cam_param_file) as f:
        cam_param_dict = {k: torch.FloatTensor(v).cuda() for k, v in json.load(f).items()}
    with open(smplx_param_file) as f:
        smplx_param_dict = {k: torch.FloatTensor(v).cuda().view(-1) for k, v in json.load(f).items()}

    # --- 步驟 2: 更新模型初始化和輸入/輸出列表 ---
    # 預先計算常數，並將它們傳遞給 Wrapper 的建構函式
    
    wrapped_model = ModelWrapper(tester.model).cuda().eval()
    
    # 更新輸入列表，現在包含了 camera 參數
    smplx_inputs_tuple = tuple(smplx_param_dict[key] for key in wrapped_model.smplx_keys)
    cam_inputs_tuple = tuple(cam_param_dict[key] for key in wrapped_model.cam_keys)
    dummy_inputs = smplx_inputs_tuple + cam_inputs_tuple
    
    # 更新輸入和輸出的名稱列表
    input_names = wrapped_model.smplx_keys + wrapped_model.cam_keys
    output_names = [
        'mean_3d', 'opacity', 'scale', 'rotation', 
        'rgb', 'mean_3d_refined', 'scale_refined','mesh_neutral_pose_wo_upsample','transform_mat_neutral_pose'
    ]
    print("PyTorch 模型與輸入準備完成。")

    # --- 步驟 3: 執行 PyTorch 模型推論 ---
    print("\n正在執行 PyTorch 模型推論...")
    with torch.no_grad():
        pytorch_outputs = wrapped_model(*dummy_inputs)
    pytorch_outputs_np = [t.cpu().numpy() for t in pytorch_outputs]
    print("PyTorch 推論完成。")

    # --- 步驟 4: 載入 ONNX 模型並執行推論 ---
    print(f"\n正在載入 ONNX 模型 '{args.onnx_path}' 並執行推論...")
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    ort_inputs = {
        input_name: input_tensor.cpu().numpy()
        for input_name, input_tensor in zip(input_names, dummy_inputs)
    }
    onnx_outputs = ort_session.run(output_names, ort_inputs)
    print("ONNX 推論完成。")

    # --- 步驟 5: 比較兩個模型的輸出 ---
    print("\n--- 輸出結果比較 ---")
    TOLERANCE = 1e-4
    all_match = True

    for i in range(len(output_names)):
        pytorch_res = pytorch_outputs_np[i]
        onnx_res = onnx_outputs[i]
        output_name = output_names[i]

        print(f"\n--- 輸出 '{output_name}' ---")
        if pytorch_res.shape != onnx_res.shape:
            print(f"   狀態: ❌ 形狀不匹配!")
            print(f"   PyTorch shape: {pytorch_res.shape}")
            print(f"   ONNX shape:    {onnx_res.shape}")
            all_match = False
            continue

        # --- 新增的誤差計算邏輯 ---
        abs_diff = np.abs(pytorch_res - onnx_res)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        num_elements = pytorch_res.size
        outlier_count = np.sum(abs_diff > TOLERANCE)
        error_ratio = outlier_count / num_elements
        
        # 根據是否有任何元素超過容忍度來判斷是否通過
        is_close = outlier_count == 0

        if is_close:
            status_icon = "✅"
            status_text = "驗證通過"
            all_match = all_match and True
        else:
            status_icon = "❌"
            status_text = "驗證失敗"
            all_match = False
        
        print(f"   狀態: {status_icon} {status_text}")
        print(f"   最大絕對誤差: {max_diff:.6g}")
        print(f"   平均絕對誤差 (MAE): {mean_diff:.6g}")
        print(f"   誤差 > {TOLERANCE} 的元素數量: {outlier_count} / {num_elements}")
        print(f"   出錯比例: {error_ratio:.4%}") # 使用百分比格式化輸出
    
    print("\n\n--- 驗證總結 ---")
    if all_match:
        print("🎉 所有輸出均在容忍度內！ONNX 模型已成功驗證。")
    else:
        print("💔 發現部分輸出不匹配。請根據上述詳細指標進行評估。")
    print(pytorch_outputs_np[0])
    print(pytorch_outputs_np[1])
    print(pytorch_outputs_np[2])
    print(pytorch_outputs_np[3])
    print(pytorch_outputs_np[4])
    print(pytorch_outputs_np[5])
    print(pytorch_outputs_np[6])
    print(pytorch_outputs_np[7])
    print(pytorch_outputs_np[8])

if __name__ == "__main__":
    main()