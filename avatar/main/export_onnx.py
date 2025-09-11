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
        smplx_inputs_tuple = inputs[:smplx_input_count]
        cam_inputs_tuple = inputs[smplx_input_count:]

        for i, key in enumerate(self.smplx_keys):
            smplx_param[key] = smplx_inputs_tuple[i]
            
        for i, key in enumerate(self.cam_keys):
            cam_param[key] = cam_inputs_tuple[i]

        # 呼叫原始模型的 human_gaussian 部分
        human_asset, _, _, _ = self.model.module.human_gaussian(smplx_param, cam_param)
        
        # 根據 module.py 的定義，human_asset 是一個字典。
        # ONNX 導出需要返回一個張量或張量的元組，因此我們提取字典中的所有張量。
        return (
            human_asset['mean_3d'],
            human_asset['opacity'],
            human_asset['scale'],
            human_asset['rotation'],
            human_asset['rgb']
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
    
    # 定義輸入和輸出的名稱 (這在之後使用 ONNX 模型時很重要)
    input_names = wrapped_model.smplx_keys + wrapped_model.cam_keys
    output_names = [
        'mean_3d', 
        'opacity', 
        'scale', 
        'rotation', 
        'rgb'
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
        opset_version=12, # 建議使用 11 或更高的版本
        export_params=True
    )
    print("模型轉換成功！")

if __name__ == "__main__":
    main()