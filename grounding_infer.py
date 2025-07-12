import light_hf_proxy
import cv2
import numpy as np
import os
from groundingdino.util.inference import load_model as dino_load_model, load_image as dino_load_image, predict as dino_predict
from PIL import Image
import torch
import logging


class GroundingDINOProcessor:
    def __init__(self, config_path, model_path, device='cuda'):
        """
        初始化 GroundingDINO 模型加载器。
        Args:
            config_path (str): GroundingDINO 配置文件的路径。
            model_path (str): GroundingDINO 模型权重的路径。
            device (str): 使用的设备 ('cpu' or 'cuda')。
        """
        logging.info(f"正在从配置 '{config_path}' 和权重 '{model_path}' 加载 GroundingDINO 模型到设备 '{device}'...")
        self.model = dino_load_model(config_path, model_path, device=device)
        self.device = device
        logging.info("GroundingDINO 模型加载成功。")

        
        self.ASPECT_RATIO_MIN = 1.0
        self.ASPECT_RATIO_MAX = 4/3
        self.AREA_RATIO_MIN = 0.08
        self.AREA_RATIO_MAX = 0.20
        
        self.DEFAULT_BOX_THRESHOLD = 0.35
        self.DEFAULT_TEXT_THRESHOLD = 0.25


    def _filter_and_crop_rois_from_boxes(self, original_image_cv, normalized_boxes_tensor):
        """
        根据给定的检测框（boxes）过滤并裁剪 ROI 区域。
        """
        if original_image_cv is None:
            logging.error("错误: _filter_and_crop_rois_from_boxes 接收到的原始图像为空。")
            return []

        img_h, img_w = original_image_cv.shape[:2]
        if img_h == 0 or img_w == 0:
            logging.error("错误: 原始图像的高度或宽度为零。")
            return []
        image_area = float(img_w * img_h)

        cropped_rois_pil = []
        
        if normalized_boxes_tensor is None or normalized_boxes_tensor.numel() == 0:
            logging.info("未检测到任何初始框。")
            return []

        normalized_boxes = normalized_boxes_tensor.cpu().numpy()

        logging.info(f"收到 {len(normalized_boxes)} 个初始检测框进行过滤。")
        for i, norm_box in enumerate(normalized_boxes):
            center_x_norm, center_y_norm, box_w_norm, box_h_norm = norm_box

            box_w_abs = box_w_norm * img_w
            box_h_abs = box_h_norm * img_h

            # 1. 检查宽高比
            aspect_ratio = box_w_abs / box_h_abs
            if not (self.ASPECT_RATIO_MIN <= aspect_ratio <= self.ASPECT_RATIO_MAX):
                logging.debug(f"  跳过框 {i}: 宽高比 ({aspect_ratio:.3f}) 超出范围 [{self.ASPECT_RATIO_MIN}, {self.ASPECT_RATIO_MAX}]。")
                continue
            
            # 2. 检查面积比例
            box_area_abs = box_w_abs * box_h_abs
            area_ratio = box_area_abs / image_area
            if not (self.AREA_RATIO_MIN <= area_ratio <= self.AREA_RATIO_MAX):
                logging.debug(f"  跳过框 {i}: 面积比例 ({area_ratio:.3f}) 超出范围 [{self.AREA_RATIO_MIN}, {self.AREA_RATIO_MAX}]。")
                continue
            
            logging.debug(f"  框 {i} 通过过滤: AR={aspect_ratio:.2f}, AreaRatio={area_ratio:.2f}")

            
            center_x_abs = center_x_norm * img_w
            center_y_abs = center_y_norm * img_h

            x1 = int(center_x_abs - box_w_abs / 2)
            y1 = int(center_y_abs - box_h_abs / 2)
            x2 = int(center_x_abs + box_w_abs / 2)
            y2 = int(center_y_abs + box_h_abs / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            if x1 >= x2 or y1 >= y2:
                logging.debug(f"  跳过框 {i}: 裁剪后坐标无效 (x1={x1}, y1={y1}, x2={x2}, y2={y2})。")
                continue
            
            cropped_image_cv = original_image_cv[y1:y2, x1:x2]

            if cropped_image_cv.size == 0:
                logging.debug(f"  跳过框 {i}: 裁剪后的图像为空。")
                continue
            
            try:
                # cropped_image_rgb = cv2.cvtColor(cropped_image_cv, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(cropped_image_cv)
                cropped_rois_pil.append(pil_image)
                logging.debug(f"  成功裁剪并转换框 {i} 为 PIL Image。")
            except Exception as e:
                logging.error(f"  处理框 {i} 时发生错误 (cv2.cvtColor 或 Image.fromarray): {e}")
            
        logging.info(f"过滤后得到 {len(cropped_rois_pil)} 个有效 ROI。")
        return cropped_rois_pil

    def get_rois_from_image(self, image_input, text_prompt: str, 
                            box_threshold: float = None, text_threshold: float = None):
        """
        从输入图像中检测、过滤并裁剪 ROI。
        """
        if box_threshold is None: box_threshold = self.DEFAULT_BOX_THRESHOLD
        if text_threshold is None: text_threshold = self.DEFAULT_TEXT_THRESHOLD

        original_cv_image = None
        dino_input_image_tensor = None

        if isinstance(image_input, str):
            cv_img, tensor_img = dino_load_image(image_input)
            if cv_img is None or tensor_img is None:
                logging.error(f"无法从路径加载图像: {image_input}")
                return []
            original_cv_image = cv_img
            dino_input_image_tensor = tensor_img
            logging.debug(f"从路径 '{image_input}' 加载图像成功。")
        elif isinstance(image_input, Image.Image): # PIL 图像对象
            pil_image_rgb = image_input.convert("RGB")
            original_cv_image = cv2.cvtColor(np.array(pil_image_rgb), cv2.COLOR_RGB2BGR)
            _, tensor_img = dino_load_image(pil_image_rgb)
            if tensor_img is None:
                logging.error("无法从 PIL.Image 转换图像为 tensor。")
                return []
            dino_input_image_tensor = tensor_img
            logging.debug("从 PIL.Image 加载图像成功。")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                original_cv_image = image_input 
                img_rgb_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
                _, tensor_img = dino_load_image(img_rgb_pil)
                if tensor_img is None:
                    logging.error("无法从 NumPy 数组转换图像为 tensor。")
                    return []
                dino_input_image_tensor = tensor_img
                logging.debug("从 NumPy 数组加载图像成功。")
            else:
                logging.error(f"不支持的 NumPy 数组形状: {image_input.shape}")
                return []
        else:
            logging.error(f"不支持的 image_input 类型: {type(image_input)}")
            return []

        if original_cv_image is None or dino_input_image_tensor is None:
            logging.error("图像未能成功加载或转换为 GroundingDINO 所需的格式。")
            return []

        logging.info(f"使用文本提示 '{text_prompt}' (box_thresh={box_threshold}, text_thresh={text_threshold}) 进行 GroundingDINO 预测...")
        
        boxes_tensor, _, _ = dino_predict(
            model=self.model,
            image=dino_input_image_tensor.to(self.device),
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        logging.info(f"GroundingDINO 预测完成。检测到 {boxes_tensor.size(0) if boxes_tensor is not None else 0} 个候选框。")
        
        # 过滤和裁剪
        cropped_pil_images = self._filter_and_crop_rois_from_boxes(original_cv_image, boxes_tensor)
        return cropped_pil_images

if __name__ == '__main__':
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    DINO_CONFIG = "/home/pjh/faiss_search/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    DINO_WEIGHTS = "/home/pjh/faiss_search/groundingdino/groundingdino_swint_ogc.pth"
    data_file = "/home/pjh/faiss_search/DataFiles_Decrypt_20250520170635/DataFiles_Decrypt"
    directory = Path(data_file)
    files = [str(file) for file in directory.iterdir() if file.suffix.lower() == '.jpeg']
    TEST_PROMPT = "whole street view"


    processor = GroundingDINOProcessor(config_path=DINO_CONFIG, model_path=DINO_WEIGHTS, device='cuda')
    
    for  TEST_IMAGE_PATH in files:
        rois = processor.get_rois_from_image(TEST_IMAGE_PATH, TEST_PROMPT)
        
        if rois:
            logging.info(f"成功从 '{TEST_IMAGE_PATH}' 提取到 {len(rois)} 个ROI。")
            output_test_dir = "/home/pjh/faiss_search/Output"
            os.makedirs(output_test_dir, exist_ok=True)
            for i, roi_img in enumerate(rois):
                file_path = Path(TEST_IMAGE_PATH)
                roi_img.save(os.path.join(output_test_dir, f"{file_path.stem}_roi_{i+1}.png"))
            logging.info(f"测试ROI已保存到 '{output_test_dir}' 目录。")
        else:
            logging.info(f"未能从 '{TEST_IMAGE_PATH}' 提取到ROI。")