import io
import os
import json
import torch
import timm
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import logging
import sys
from torch.nn import functional as F
import asyncio
import aiohttp


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageRetrieval:
    def __init__(self, model_path, config_path, device='cpu'):
        self.device = device
        self.model = None
        self.transforms = None
        self.index = None
        self.metadata = None
        self.dimension = None

        logging.info(f"使用的设备: {self.device}")

        # 加载模型
        try:
            if not os.path.exists(model_path):
                logging.error(f"模型文件未找到: {model_path}。")
                raise FileNotFoundError(f"模型文件未找到: {model_path}")
            logging.info(f"从 {model_path} 加载模型...")
            self.model = torch.load(model_path, map_location=self.device)
            self.model = self.model.eval().to(self.device)
            logging.info("模型加载成功。")
        except Exception as e:
            logging.error(f"加载模型时出错: {e}")
            raise

        # 加载图像转换配置
        try:
            if not os.path.exists(config_path):
                logging.error(f"配置文件未找到: {config_path}。")
                raise FileNotFoundError(f"配置文件未找到: {config_path}")

            with open(config_path) as f:
                data_config_full = json.load(f)
            
            data_config = data_config_full.get('pretrained_cfg', data_config_full)
            required_keys = ['input_size', 'interpolation', 'mean', 'std', 'crop_pct']
            config_params = {}
            for key in required_keys:
                if key not in data_config:
                    logging.warning(f"配置中未找到键 '{key}'。将尝试使用timm的默认值。")
                else:
                    config_params[key] = data_config[key]
            
            if 'crop_mode' in data_config:
                 config_params['crop_mode'] = data_config['crop_mode']
            
            inp_size = config_params.get('input_size')
            if isinstance(inp_size, (list, tuple)) and len(inp_size) == 2:
                config_params['input_size'] = (3, inp_size[0], inp_size[1])
            elif isinstance(inp_size, (list, tuple)) and len(inp_size) == 3:
                config_params['input_size'] = tuple(inp_size)
            else:
                logging.warning(f"输入尺寸格式不符合预期: {inp_size}。将依赖timm的默认值。")
                if 'input_size' in config_params: del config_params['input_size']

            self.transforms = timm.data.create_transform(**config_params, is_training=False)
            logging.info("图像转换加载成功。")
        except FileNotFoundError:
            raise
        except Exception as e:
            logging.error(f"加载图像转换时出错: {e}")
            raise

    def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(feature)
        return feature / norm if norm != 0 else feature

    def extract_feature(self, img_input) -> np.ndarray:
        try:
            if isinstance(img_input, str):
                img = Image.open(img_input).convert('RGB')
            elif isinstance(img_input, Image.Image):
                img = img_input.convert('RGB')
            else:
                img = Image.fromarray(img_input).convert('RGB')

            with torch.no_grad():
                feature_tensor = self.model(self.transforms(img).unsqueeze(0).to(self.device))
            feature = feature_tensor.squeeze().cpu().numpy()
            return self._normalize_feature(feature)
        except Exception as e:
            logging.error(f"从 {type(img_input)} 提取特征时出错: {e}")
            raise
    
    
    def extract_features_in_batches(self, image_source_list: list, source_type: str, batch_size: int = 32):
        """
        批量处理图像以提取特征，支持从本地路径或已加载的PIL图像列表。
        """
        all_features = []
        valid_sources_in_order = []
        
        for i in tqdm(range(0, len(image_source_list), batch_size), desc=f"批量提取特征中 (来源: {source_type})"):
            batch_sources = image_source_list[i:i + batch_size]
            batch_tensors = []
            current_batch_valid_sources = []
            
            for source in batch_sources:
                try:
                    # 根据来源类型处理图像
                    if source_type == 'path':
                        img = Image.open(source).convert("RGB")
                    elif source_type == 'pil':
                        img = source # source 本身就是 PIL image
                    else:
                        continue # 不支持的类型
                        
                    transformed_img_tensor = self.transforms(img)
                    batch_tensors.append(transformed_img_tensor)
                    current_batch_valid_sources.append(source)
                except Exception as e:
                    logging.warning(f"跳过源 '{str(source)[:100]}'，因加载或转换时出错: {e}")
            
            if not batch_tensors:
                continue

            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                feature_tensor_batch = self.model(batch_tensor)
                normalized_tensor_batch = F.normalize(feature_tensor_batch, p=2, dim=1)

            features_np_batch = normalized_tensor_batch.cpu().numpy()
            all_features.extend(features_np_batch)
            valid_sources_in_order.extend(current_batch_valid_sources)
            
        return all_features, valid_sources_in_order
    
    
    def build_index(self, data_root: str, index_base_name: str, index_dir: str, batch_size: int = 64):
        """
        通过批量提取特征的方式扫描目录并构建FAISS索引。
        Args:
            data_root (str): 包含图片的根目录。
            index_base_name (str): 索引文件的基础名称。
            index_dir (str): 存储索引文件的目录。
            batch_size (int): 特征提取时使用的批处理大小。
        """
        if not os.path.isdir(data_root):
            logging.error(f"数据源目录未找到: {data_root}")
            self.metadata = {'image_paths': []} 
            return self

        
        image_paths = []
        logging.info(f"开始从 {data_root} 递归扫描图片文件")
        valid_extensions = ('.jpg', '.jpeg', '.png',)
        for root, _, files in os.walk(data_root):
            for file_name in files:
                if file_name.lower().endswith(valid_extensions):
                    abs_path = os.path.abspath(os.path.join(root, file_name))
                    image_paths.append(abs_path)
        
        if not image_paths:
            logging.warning(f"在 {data_root} 中未找到图片。索引将为空。")
            self.metadata = {'image_paths': []}
            return self

        logging.info(f"找到 {len(image_paths)} 张图片。开始使用批大小 {batch_size} 进行特征提取。")

        
        features_list, valid_image_paths = self.extract_features_in_batches(image_paths, batch_size=batch_size, source_type='path')

        if not features_list:
            logging.error("未能提取任何特征。无法构建索引。")
            self.metadata = {'image_paths': []}
            return self
            
        features_np = np.array(features_list).astype('float32')
        
        self.dimension = features_np.shape[1]
        logging.info(f"使用维度 {self.dimension} 构建FAISS索引...")
        self.index = faiss.index_factory(self.dimension, 'HNSW64', faiss.METRIC_L2)
        self.index.add(features_np)
        
        # 使用有效图片的路径更新元数据
        self.metadata = {'image_paths': valid_image_paths}
        
        logging.info(f"索引构建完成。共索引图片数量: {self.index.ntotal}")
        self.save_index(index_base_name, index_dir)
        return self
    
    
    def build_index_from_urls(self, url_list_txt: str, index_base_name: str, index_dir: str, 
                              batch_size: int = 64, download_concurrency: int = 100):
        """
        通过异步下载URL列表中的图片并批量提取特征来构建FAISS索引。
        
        Args:
            url_list_txt (str): 包含图片URL列表的文本文件路径，每行一个URL。
            index_base_name (str): 索引文件的基础名称。
            index_dir (str): 存储索引文件的目录。
            batch_size (int): 特征提取时使用的批处理大小。
            download_concurrency (int): 并发下载的协程数量。
        """
        # 1. 从 .txt 文件读取所有 URL
        try:
            with open(url_list_txt, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            if not urls:
                logging.warning(f"URL 列表文件 '{url_list_txt}' 为空或未包含有效URL。")
                return self
            logging.info(msg=f"从 '{url_list_txt}' 读取到 {len(urls)} 个 URL。")
        except FileNotFoundError:
            logging.error(f"URL 列表文件未找到: {url_list_txt}")
            return self

        # 2. 运行异步下载和处理流程
        async def main_async_pipeline():
            # aiohttp.ClientSession 用于复用连接，提高效率
            # TCPConnector 用于限制并发连接数，防止对服务器造成过大压力
            connector = aiohttp.TCPConnector(limit=download_concurrency)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = []
                # 为每个 URL 创建一个下载任务
                for url in urls:
                    tasks.append(asyncio.create_task(self._download_and_decode_image(session, url)))
                
                # 使用 asyncio.gather 并发执行所有下载任务
                # tqdm 用于显示下载进度
                results = []
                for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="异步下载图片中"):
                    result = await f
                    if result: # 只添加成功下载和解码的结果
                        results.append(result)
                return results

        # 运行主异步流程
        logging.info(f"启动异步下载流程，并发数: {download_concurrency}...")
        # downloaded_results 是一个元组列表 [(url1, pil_image1), (url2, pil_image2), ...]
        downloaded_results = asyncio.run(main_async_pipeline())

        if not downloaded_results:
            logging.error("未能成功下载和解码任何图片。无法构建索引。")
            return self
            
        logging.info(f"成功下载并解码 {len(downloaded_results)} 张图片。")

        # 3. 准备进行批量特征提取
        # 将下载结果解包为两个列表：urls 和 pil_images
        valid_urls, pil_images = zip(*downloaded_results)
        
        # 调用批量提取方法，这次源类型是 'pil'
        features_list, final_valid_urls = self.extract_features_in_batches(
            list(pil_images), source_type='pil', batch_size=batch_size
        )

        # 4. 构建 FAISS 索引 (与 build_index 方法的后半部分相同)
        if not features_list:
            logging.error("未能从下载的图片中提取任何特征。")
            return self

        features_np = np.array(features_list).astype('float32')
        self.dimension = features_np.shape[1]
        logging.info(f"使用维度 {self.dimension} 构建FAISS索引...")
        self.index = faiss.index_factory(self.dimension, 'HNSW64', faiss.METRIC_L2)
        self.index.add(features_np)
        
        # --- 重要：元数据现在存储的是 URL ---
        self.metadata = {'image_paths': list(final_valid_urls)}
        
        logging.info(f"索引构建完成。共索引图片数量: {self.index.ntotal}")
        self.save_index(index_base_name, index_dir)
        return self

    async def _download_and_decode_image(self, session: aiohttp.ClientSession, url: str):
        """
        一个异步的辅助函数，用于下载单个图片并将其解码为 PIL.Image 对象。
        """
        try:
            async with session.get(url, timeout=30) as response:
                response.raise_for_status() 
                image_bytes = await response.read()
                # 在内存中从字节流解码图像
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                return url, pil_image
        except asyncio.TimeoutError:
            logging.warning(f"下载超时: {url}")
        except aiohttp.ClientError as e:
            logging.warning(f"下载时发生客户端错误: {url}, 错误: {e}")
        except Exception as e:
            logging.warning(f"处理图片URL时发生未知错误: {url}, 错误: {e}")
        return None

    
    def get_index_paths(self, index_base_name: str, index_dir: str):
        faiss_index_path = os.path.join(index_dir, f"{index_base_name}.faiss")
        metadata_path = os.path.join(index_dir, f"{index_base_name}.pkl")
        return faiss_index_path, metadata_path

    def save_index(self, index_base_name: str, index_dir: str):
        if self.index is None or self.metadata is None:
            logging.warning("索引或元数据未构建。无需保存。")
            return

        os.makedirs(index_dir, exist_ok=True)
        faiss_index_path, metadata_path = self.get_index_paths(index_base_name, index_dir)

        try:
            logging.info(f"保存FAISS索引到 {faiss_index_path}")
            faiss.write_index(self.index, faiss_index_path)
            logging.info(f"保存元数据到 {metadata_path}")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logging.error(f"保存索引/元数据时出错: {e}")

    def load_index(self, index_base_name: str, index_dir: str):
        faiss_index_path, metadata_path = self.get_index_paths(index_base_name, index_dir)

        if not os.path.exists(faiss_index_path) or not os.path.exists(metadata_path):
            logging.warning(f"索引文件 ({faiss_index_path}) 或元数据文件 ({metadata_path}) 未找到。")
            return False

        try:
            logging.info(f"从 {faiss_index_path} 加载FAISS索引...")
            self.index = faiss.read_index(faiss_index_path)
            logging.info(f"从 {metadata_path} 加载元数据...")
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            self.dimension = self.index.d
            logging.info(f"索引 (包含 {self.index.ntotal} 个向量) 和元数据加载成功。")
            if self.metadata and 'image_paths' in self.metadata:
                self.metadata['image_paths'] = [os.path.abspath(os.path.normpath(p)) for p in self.metadata['image_paths']]
            return True
        except Exception as e:
            logging.error(f"加载索引/元数据时出错: {e}")
            self.index = None
            self.metadata = None
            return False

    def search_similar_images(self, query_image_input, k: int = 5):
        if self.index is None or self.metadata is None or 'image_paths' not in self.metadata:
            logging.error("索引未加载、元数据不完整或未构建。")
            return []
        if self.index.ntotal == 0:
            logging.warning("索引为空。无法执行搜索。")
            return []

        try:
            query_feature = self.extract_feature(query_image_input)
        except Exception as e:
            logging.error(f"提取查询图像特征失败: {e}")
            return []
            
        query_feature_np = query_feature.reshape(1, -1).astype('float32')
        
        distances, indices = self.index.search(query_feature_np, k)
        
        results = []
        if not indices.size or indices[0][0] == -1:
            logging.info("在索引中没有找到相似的图片。")
            return results

        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1: continue

            dist_sq = distances[0][i]
            similarity = float(1 - (dist_sq / 2))
            similarity = max(0.0, min(1.0, similarity))

            if idx < len(self.metadata['image_paths']):
                image_path = self.metadata['image_paths'][idx]

                results.append({
                    'rank': i + 1,
                    'similarity': similarity,
                    'image_path': image_path
                })
            else:
                logging.warning(f"找到的索引 {idx} 超出元数据图像路径列表的范围 (长度: {len(self.metadata['image_paths'])})。")
        return results
    
    def search_similar_images_in_directory(self, query_image_input, target_directory: str, k: int = 5, 
                                           similarity_threshold: float = 0.8, k_initial_search_multiplier: int = 20):
        """
        在指定目录中搜索与查询图像相似且超过特定相似度阈值的图像。

        参数:
            query_image_input: 查询图片 (路径 str, PIL.Image, 或 NumPy 数组)。
            target_directory (str): 要在其中搜索图像的目标目录的路径。
            k (int): 返回的最大结果数量 (Top-K)。
            similarity_threshold (float): 相似度的最小阈值 (0.0 到 1.0)。
            k_initial_search_multiplier (int): 初始FAISS搜索时k值的乘数，以获得更广泛的候选池。

        返回:
            dict: 包含搜索状态、消息和结果列表的字典。
                  例如: {"status": "success/info/error", "message": "...", "results": [...] }
                  每个结果是一个字典: {'rank': ..., 'similarity': ..., 'image_path': ...}
        """
        if self.index is None or self.metadata is None or 'image_paths' not in self.metadata:
            msg = "索引未加载、元数据不完整或未构建。"
            logging.error(msg)
            return {"status": "error", "message": msg, "results": []}
        if self.index.ntotal == 0:
            msg = "索引为空。无法执行搜索。"
            logging.warning(msg)
            return {"status": "warning", "message": msg, "results": []}

        abs_norm_target_dir = os.path.abspath(os.path.normpath(target_directory))

        # 检查目标目录或其子目录中是否有任何已索引的图像
        found_any_in_dir_scope = False
        for indexed_img_path in self.metadata['image_paths']:
            image_parent_dir = os.path.dirname(indexed_img_path)
            # 检查图像的父目录是否是目标目录，或者是目标目录的子目录
            if image_parent_dir == abs_norm_target_dir or \
               image_parent_dir.startswith(abs_norm_target_dir + os.sep):
                found_any_in_dir_scope = True
                break
        
        if not found_any_in_dir_scope:
            msg = (f"目标目录 '{target_directory}' (解析为 '{abs_norm_target_dir}') "
                   f"或其子目录中没有找到已索引的图像。请确认该目录下的图片已被索引，并且路径输入正确。")
            logging.info(msg)
            return {"status": "info", "message": msg, "results": []}

        # 提取查询图像的特征
        try:
            query_feature = self.extract_feature(query_image_input)
        except Exception as e:
            msg = f"提取查询图像特征失败: {e}"
            logging.error(msg)
            return {"status": "error", "message": msg, "results": []}
            
        query_feature_np = query_feature.reshape(1, -1).astype('float32')

        num_to_search_initially = min(self.index.ntotal, max(k * k_initial_search_multiplier, 100))
        
        distances_sq, indices = self.index.search(query_feature_np, num_to_search_initially)

        candidate_results = []
        if indices.size > 0 and indices[0][0] != -1:
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx == -1: continue

                # 确保索引在元数据范围内
                if not (0 <= idx < len(self.metadata['image_paths'])):
                    logging.warning(f"搜索返回的索引 {idx} 超出元数据范围。")
                    continue
                
                image_path = self.metadata['image_paths'][idx]

                image_parent_dir_loop = os.path.dirname(image_path)
                is_in_target_dir = False
                if image_parent_dir_loop == abs_norm_target_dir or \
                   image_parent_dir_loop.startswith(abs_norm_target_dir + os.sep):
                    is_in_target_dir = True

                if not is_in_target_dir:
                    continue

                dist_sq = distances_sq[0][i]
                similarity = float(1-(dist_sq/2))
                similarity = max(0.0, min(1.0, similarity))

                if similarity >= similarity_threshold:
                    candidate_results.append({
                        'similarity': similarity,
                        'image_path': image_path
                    })

        candidate_results.sort(key=lambda x: x['similarity'], reverse=True)
        

        final_results = []
        for rank, res_item in enumerate(candidate_results[:k]):
            final_results.append({
                'rank': rank + 1,
                'similarity': res_item['similarity'],
                'image_path': res_item['image_path']
            })

        if not final_results:
            found_in_dir_before_thresholding = any(
                os.path.dirname(self.metadata['image_paths'][indices[0][i]]) == abs_norm_target_dir or \
                os.path.dirname(self.metadata['image_paths'][indices[0][i]]).startswith(abs_norm_target_dir + os.sep)
                for i in range(len(indices[0])) if indices[0][i] != -1 and (0 <= indices[0][i] < len(self.metadata['image_paths']))
            )
            if not found_in_dir_before_thresholding:
                 message = f"在目录 '{target_directory}' (解析为 '{abs_norm_target_dir}') 的已索引图片中，没有与查询图片相似的图片"
            else:
                message = (f"在目录 '{target_directory}' (解析为 '{abs_norm_target_dir}') 中找到了一些相似图片，"
                           f"但没有一张的相似度达到或超过 {similarity_threshold:.2f}。")
            logging.info(message)
            return {"status": "info", "message": message, "results": []}
        
        message = (f"在目录 '{target_directory}' (解析为 '{abs_norm_target_dir}') 中找到 {len(final_results)} 张相似图片 "
                   f"(相似度 >= {similarity_threshold:.2f})。")
        logging.info(message)
        return {"status": "success", "message": message, "results": final_results}
    
    def search_globally(self, query_image_input, k: int = 10, similarity_threshold: float = 0.8):
        """
        在整个索引中执行全局相似性搜索，返回包含 rank 的结构化结果。
        """
        if self.index is None or self.metadata is None or 'image_paths' not in self.metadata:
            return {"status": "error", "message": "索引未加载、元数据不完整或未构建。", "results": []}
        if self.index.ntotal == 0:
            return {"status": "warning", "message": "索引为空，无法执行搜索。", "results": []}

        try:
            query_feature = self.extract_feature(query_image_input)
        except Exception as e:
            return {"status": "error", "message": f"提取查询图像特征失败: {e}", "results": []}
            
        query_feature_np = query_feature.reshape(1, -1).astype('float32')
        
        num_to_search = min(self.index.ntotal, k * 5 if k > 0 else 50) 
        distances_sq, indices = self.index.search(query_feature_np, num_to_search)
        
        candidate_results = []
        if not indices.size or indices[0][0] == -1:
            return {"status": "success", "message": "在索引中没有找到相似的图片。", "results": []}

        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1: continue

            if not (0 <= idx < len(self.metadata['image_paths'])):
                continue

            dist_sq = distances_sq[0][i]
            similarity = float(1 - (dist_sq / 2))
            
            if similarity >= similarity_threshold:
                candidate_results.append({
                    'similarity': max(0.0, min(1.0, similarity)),
                    'image_path': self.metadata['image_paths'][idx]
                })
        
        candidate_results.sort(key=lambda x: x['similarity'], reverse=True)

        final_results = candidate_results[:k]

        for rank, item in enumerate(final_results):
            item['rank'] = rank + 1
            
        return {
            "status": "success",
            "message": f"全局搜索完成，找到 {len(final_results)} 个符合条件的结果。",
            "results": final_results
        }
    

if __name__ == "__main__":
    MODEL_PATH = "/home/pjh/faiss_search/model_and_index/mobilenetv4.pth"
    CONFIG_PATH = "/home/pjh/faiss_search/model_and_index/config.json"
    DATA_ROOT = "/home/pjh/faiss_search/Output"
    INDEX_DIR = "/home/pjh/faiss_search/model_and_index"
    INDEX_NAME = "Crop_Decrypt_index"
    
    image_retrieval = ImageRetrieval(model_path=MODEL_PATH, config_path=CONFIG_PATH)
    
    image_retrieval.build_index(data_root=DATA_ROOT, index_base_name=INDEX_NAME, index_dir=INDEX_DIR)
    
    
    