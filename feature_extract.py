import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import json
import logging
import os
import pickle
from typing import List, Dict, Tuple
from tqdm import tqdm
from queue import Queue
from threading import Thread
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractor:
    """图像特征提取器"""
    def __init__(self, model_path: str, config_path: str, device: str = None):
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.transforms = None
        
        # 加载模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        logging.info(f"从 {model_path} 加载模型到 {self.device}...")
        self.model = torch.load(model_path, map_location=self.device)
        self.model = self.model.eval().to(self.device)
        logging.info("模型加载成功")
        
        # 加载配置和转换
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
        with open(config_path) as f:
            data_config_full = json.load(f)
        
        data_config = data_config_full.get('pretrained_cfg', data_config_full)
        config_params = {}
        
        # 提取必要的配置参数
        for key in ['input_size', 'interpolation', 'mean', 'std', 'crop_pct']:
            if key in data_config:
                config_params[key] = data_config[key]
        
        if 'crop_mode' in data_config:
            config_params['crop_mode'] = data_config['crop_mode']
        
        # 处理input_size格式
        inp_size = config_params.get('input_size')
        if isinstance(inp_size, (list, tuple)) and len(inp_size) == 2:
            config_params['input_size'] = (3, inp_size[0], inp_size[1])
        elif isinstance(inp_size, (list, tuple)) and len(inp_size) == 3:
            config_params['input_size'] = tuple(inp_size)
        
        self.transforms = timm.data.create_transform(**config_params, is_training=False)
        logging.info("图像转换配置加载成功")
    
    def extract_feature(self, image_path: str) -> np.ndarray:
        """从图像文件提取特征向量"""
        try:
            img = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
                feature_tensor = self.model(img_tensor)
                feature_tensor = F.normalize(feature_tensor, p=2, dim=1)
                
            return feature_tensor.squeeze().cpu().numpy()
        except Exception as e:
            logging.error(f"提取特征失败 {image_path}: {e}")
            return None
    
    def extract_features_batch(self, image_paths: List[str], batch_size: int = 256) -> List[Tuple[np.ndarray, str]]:
        """批量提取特征"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_paths = []
            
            # 加载和预处理批次图像
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transforms(img)
                    batch_tensors.append(img_tensor)
                    valid_paths.append(path)
                except Exception as e:
                    logging.error(f"加载图像失败 {path}: {e}")
                    continue
            
            if not batch_tensors:
                continue
            
            # 批量推理
            try:
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                with torch.no_grad():
                    features = self.model(batch_tensor)
                    features = F.normalize(features, p=2, dim=1)
                
                features_np = features.cpu().numpy()
                
                # 清理缓存和显存
                del features
                del batch_tensor
                
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                # 收集结果
                for feature, path in zip(features_np, valid_paths):
                    results.append((feature, path))
                    
            except Exception as e:
                logging.error(f"批量推理失败: {e}")
                # 如果批量失败，尝试单个处理
                for path in valid_paths:
                    feature = self.extract_feature(path)
                    if feature is not None:
                        results.append((feature, path))
        
        return results

class IncrementalStorage:
    """增量存储特征和元数据"""
    def __init__(self, feature_path: str, metadata_path: str):
        self.feature_path = feature_path
        self.metadata_path = metadata_path
        
        # 确保目录存在
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # 加载现有数据
        self.features = self._load_features()
        self.metadata = self._load_metadata()
    
    def _load_features(self) -> List[np.ndarray]:
        """加载现有特征"""
        if os.path.exists(self.feature_path):
            try:
                features = np.load(self.feature_path)
                return features.tolist()
            except Exception as e:
                logging.warning(f"加载特征文件失败: {e}")
        return []
    
    def _load_metadata(self) -> Dict:
        """加载现有元数据，保持原始格式"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    # 确保格式正确
                    if isinstance(metadata, dict) and 'image_paths' in metadata:
                        return metadata
            except Exception as e:
                logging.warning(f"加载元数据文件失败: {e}")
        return {'image_paths': []}
    
    def add(self, feature: np.ndarray, bucket_key: str):
        """添加特征和对应的bucket+key路径"""
        self.features.append(feature)
        self.metadata['image_paths'].append(bucket_key)
    
    def add_batch(self, features_and_keys: List[Tuple[np.ndarray, str]]):
        """批量添加特征和路径"""
        for feature, bucket_key in features_and_keys:
            self.features.append(feature)
            self.metadata['image_paths'].append(bucket_key)
    
    def save(self):
        """保存当前的特征和元数据到文件"""
        try:
            # 保存特征
            if self.features:
                features_array = np.array(self.features, dtype=np.float32)
                np.save(self.feature_path, features_array)
                logging.info(f"保存了 {len(self.features)} 个特征到 {self.feature_path}")
            
            # 保存元数据
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
                logging.info(f"保存了 {len(self.metadata['image_paths'])} 条元数据到 {self.metadata_path}")
        except Exception as e:
            logging.error(f"保存数据失败: {e}")
            raise
    
    def __len__(self):
        return len(self.features)


class AsyncFeatureProcessor:
    
    def __init__(self, model_path: str, config_path: str, 
                 feature_save_path: str, metadata_save_path: str,
                 batch_size: int = 256, batch_save_interval=10000, device: str = None):
        self.batch_size = batch_size
        self.extractor = FeatureExtractor(model_path, config_path, device)
        self.storage = IncrementalStorage(feature_save_path, metadata_save_path)
        self.batch_save_interval = batch_save_interval
        
        # 任务队列和结果队列
        self.task_queue = Queue()
        self.processing = False
        self.worker_thread = None
        
    def _worker(self):
        """后台工作线程 - 处理特征提取任务"""
        while self.processing or not self.task_queue.empty():
            batch_tasks = []
            
            while len(batch_tasks) < self.batch_size and not self.task_queue.empty():
                try:
                    task = self.task_queue.get(timeout=0.1)
                    batch_tasks.append(task)
                except:
                    break
            
            if not batch_tasks:
                if self.processing:
                    time.sleep(0.1)
                continue
            
            # 提取特征
            image_paths = [task[0] for task in batch_tasks]
            bucket_keys_map = {task[0]: task[1] for task in batch_tasks}
            
            try:
                # 批量提取特征
                results = self.extractor.extract_features_batch(image_paths, self.batch_size)
                
                # 准备存储数据
                features_to_save = []
                for feature, path in results:
                    if path in bucket_keys_map:
                        features_to_save.append((feature, bucket_keys_map[path]))
                
                # 批量添加到存储
                if features_to_save:
                    logging.info(f"添加了{len(features_to_save)}个特征")
                    self.storage.add_batch(features_to_save)
                    
                    if len(self.storage) % self.batch_save_interval == 0:
                        self.storage.save()
                        
            except Exception as e:
                logging.error(f"批量特征提取失败: {e}")
    
    def start(self):
        """启动异步处理"""
        if not self.processing:
            self.processing = True
            self.worker_thread = Thread(target=self._worker)
            self.worker_thread.start()
            logging.info("异步特征处理器已启动")
    
    def add_files(self, file_info_list: List[Tuple[str, str]]):
        """添加文件到处理队列"""
        for file_info in file_info_list:
            self.task_queue.put(file_info)
        logging.info(f"添加了 {len(file_info_list)} 个文件到处理队列")
    
    def stop(self):
        """停止处理并等待完成"""
        logging.info("等待所有特征提取任务完成...")
        self.processing = False
        
        if self.worker_thread:
            self.worker_thread.join()
        
        # 最终保存
        self.storage.save()
        logging.info(f"特征提取完成，共处理 {len(self.storage)} 个文件")
        
    def get_queue_size(self):
        """获取待处理队列大小"""
        return self.task_queue.qsize()