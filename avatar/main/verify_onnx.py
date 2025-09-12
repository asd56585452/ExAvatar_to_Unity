import torch
import json
import argparse
import os.path as osp
import numpy as np
import onnxruntime # 載入 ONNX 執行環境

# 假設您的 config, base, model, smpl_x 模組都在可導入的路徑中
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from model import get_model

# 重新使用您在 export_onnx.py 中定義的 ModelWrapper
# 這樣我們就可以用同樣的方式處理輸入和輸出
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.smplx_keys = [
             'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 
            'lhand_pose', 'rhand_pose', 'expr'
        ]
        self.cam_keys = []

    def forward(self, *inputs):
        smplx_param = {}
        cam_param = {}
        smplx_input_count = len(self.smplx_keys)
        smplx_inputs_tuple = inputs[:smplx_input_count]
        cam_inputs_tuple = inputs[smplx_input_count:]

        for i, key in enumerate(self.smplx_keys):
            smplx_param[key] = smplx_inputs_tuple[i]
        for i, key in enumerate(self.cam_keys):
            cam_param[key] = cam_inputs_tuple[i]

        tri_feat = self.model.module.human_gaussian.extract_tri_feature()
        geo_feat = self.model.module.human_gaussian.geo_net(tri_feat)
        mean_offset = self.model.module.human_gaussian.mean_offset_net(geo_feat)
        scale = self.model.module.human_gaussian.scale_net(geo_feat)
        rgb = self.model.module.human_gaussian.rgb_net(tri_feat)
        mean_3d_offset = mean_offset

        mean_offset_offset, scale_offset = self.model.module.human_gaussian.forward_geo_network(tri_feat, smplx_param)
        scale_refined = torch.exp(scale + scale_offset).repeat(1, 3)
        scale = torch.exp(scale).repeat(1, 3)
        mean_combined_offset, _ = self.model.module.human_gaussian.get_mean_offset_offset(smplx_param, mean_offset_offset)
        mean_3d_refined_offset = mean_3d_offset + mean_combined_offset

        smplx_expr_offset = (smplx_param['expr'][None,None,:] * self.model.module.human_gaussian.expr_dirs).sum(2)
        mean_3d_offset = mean_3d_offset + smplx_expr_offset
        mean_3d_refined_offset = mean_3d_refined_offset + smplx_expr_offset

        rgb = (torch.tanh(rgb) + 1) / 2
        
        return (
            mean_3d_offset,
            scale,
            rgb,
            mean_3d_refined_offset,
            scale_refined
        )

def main():
    # --- 與 export_onnx.py 完全相同的參數設定 ---
    parser = argparse.ArgumentParser(description="Verify ONNX model against PyTorch model")
    parser.add_argument('--subject_id', type=str, required=True, help="Subject ID")
    parser.add_argument('--test_epoch', type=str, required=True, help="Model checkpoint epoch")
    parser.add_argument('--motion_path', type=str, required=True, help="Path to motion data")
    parser.add_argument('--onnx_path', type=str, default='human_gaussian_model.onnx', help="Path to the ONNX model to verify")
    args = parser.parse_args()

    cfg.set_args(args.subject_id)

    # --- 步驟 1: 載入原始 PyTorch 模型並準備範例輸入 (與 export_onnx.py 相同) ---
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

    frame_idx = 100
    cam_param_file = osp.join(args.motion_path, 'cam_params', f'{frame_idx}.json')
    smplx_param_file = osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', f'{frame_idx}.json')

    with open(cam_param_file) as f:
        cam_param_dict = {k: torch.FloatTensor(v).cuda() for k, v in json.load(f).items()}
    with open(smplx_param_file) as f:
        smplx_param_dict = {k: torch.FloatTensor(v).cuda().view(-1) for k, v in json.load(f).items()}

    wrapped_model = ModelWrapper(tester.model).cuda().eval()
    print(smplx_param_dict.keys.__str__)
    smplx_inputs_tuple = tuple(smplx_param_dict[key] for key in wrapped_model.smplx_keys)
    cam_inputs_tuple = tuple(cam_param_dict[key] for key in wrapped_model.cam_keys)
    dummy_inputs = smplx_inputs_tuple + cam_inputs_tuple
    
    input_names = wrapped_model.smplx_keys + wrapped_model.cam_keys
    output_names = [
        'mean_3d_offset', 'scale', 'rgb', 
        'mean_3d_refined_offset', 'scale_refined'
    ]
    print("PyTorch 模型與輸入準備完成。")

    # --- 步驟 2: 執行 PyTorch 模型推論 ---
    print("\n正在執行 PyTorch 模型推論...")
    with torch.no_grad():
        pytorch_outputs = wrapped_model(*dummy_inputs)
    # 將 PyTorch 輸出轉為 NumPy 陣列以便比較
    pytorch_outputs_np = [t.cpu().numpy() for t in pytorch_outputs]
    print("PyTorch 推論完成。")

    # --- 步驟 3: 載入 ONNX 模型並執行推論 ---
    print(f"\n正在載入 ONNX 模型 '{args.onnx_path}' 並執行推論...")
    ort_session = onnxruntime.InferenceSession(args.onnx_path)

    # 準備 ONNX Runtime 的輸入格式 (字典，key 為輸入名稱，value 為 NumPy 陣列)
    ort_inputs = {
        input_name: input_tensor.cpu().numpy()
        for input_name, input_tensor in zip(input_names, dummy_inputs)
    }

    # 執行推論
    onnx_outputs = ort_session.run(output_names, ort_inputs)
    print("ONNX 推論完成。")

    # --- 步驟 4: 比較兩個模型的輸出 ---
    print("\n--- 輸出結果比較 ---")
    TOLERANCE = 1e-5  # 設定一個合理的誤差容忍度
    all_match = True

    for i in range(len(output_names)):
        pytorch_res = pytorch_outputs_np[i]
        onnx_res = onnx_outputs[i]
        output_name = output_names[i]

        # 檢查形狀是否一致
        if pytorch_res.shape != onnx_res.shape:
            print(f"❌ 輸出 '{output_name}' 的形狀不匹配!")
            print(f"   PyTorch shape: {pytorch_res.shape}")
            print(f"   ONNX shape:    {onnx_res.shape}")
            all_match = False
            continue

        # 檢查數值是否接近 (使用 allclose 處理浮點數誤差)
        is_close = np.allclose(pytorch_res, onnx_res, atol=TOLERANCE)
        max_diff = np.max(np.abs(pytorch_res - onnx_res))

        if is_close:
            print(f"✅ 輸出 '{output_name}' 驗證通過。")
            print(f"   最大絕對誤差: {max_diff:.6g} (在容忍度 {TOLERANCE} 內)")
        else:
            print(f"❌ 輸出 '{output_name}' 驗證失敗!")
            print(f"   最大絕對誤差: {max_diff:.6g} (超過容忍度 {TOLERANCE})")
            all_match = False
    
    print("\n--- 驗證總結 ---")
    if all_match:
        print("🎉 所有輸出均匹配！ONNX 模型已成功驗證。")
    else:
        print("💔 發現輸出不匹配。請檢查模型轉換過程中的警告或誤差。")
    print(onnx_outputs[0])

if __name__ == "__main__":
    main()