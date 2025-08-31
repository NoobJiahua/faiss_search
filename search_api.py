import asyncio
import os
import logging
import io
import sys
from datetime import datetime
from typing import List
from unittest import result
import cv2 
import numpy as np
from PIL import Image
import base64
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import torch
from faiss_search import ImageRetrieval
from grounding_infer import GroundingDINOProcessor

MODEL_PATH = os.getenv('MODEL_PATH', "/home/fuxin/faiss_search/model_and_index/mobilenetv4.pth")
CONFIG_PATH = os.getenv('CONFIG_PATH', "/home/fuxin/faiss_search/model_and_index/config.json")
DATA_ROOT = os.getenv('DATA_ROOT', "/home/fuxin/faiss_search/Output")
# INDEX_DIR = os.getenv('INDEX_DIR', "/home/fuxin/faiss_search/model_and_index")  #修改为你的索引文件夹路径
INDEX_DIR = os.getenv('INDEX_DIR', "/home/fuxin/AiPatrol/saved_npy_for_index")  #修改为你的索引文件夹路径
INDEX_NAME = os.getenv('INDEX_NAME', None)


DINO_CONFIG_PATH = os.getenv('DINO_CONFIG_PATH', "/home/fuxin/faiss_search/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DINO_MODEL_PATH = os.getenv('DINO_MODEL_PATH', "/home/fuxin/faiss_search/groundingdino/groundingdino_swint_ogc.pth")
DEFAULT_DINO_TEXT_PROMPT = os.getenv('DEFAULT_DINO_TEXT_PROMPT', "whole street view")

DEVICE = os.getenv('DEVICE', "cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


retrieval_system_instance = None
dino_processor_instance = None
api_lock = asyncio.Lock()
app = FastAPI()

def initialize_faiss_retrieval_system():
    global retrieval_system_instance
    if retrieval_system_instance is None:
        logger.info("正在为 API 初始化 FAISS ImageRetrieval 系统...")
        try:
            if not os.path.exists(MODEL_PATH):
                logger.error(f"FAISS 特征提取模型文件 {MODEL_PATH} 未找到。系统无法初始化。")
                return False
            
            retrieval_system_instance = ImageRetrieval(MODEL_PATH, CONFIG_PATH, device=DEVICE)
            if not retrieval_system_instance.load_index(INDEX_NAME, INDEX_DIR):
                logger.warning(f"未能加载默认索引 '{INDEX_NAME}'。API 将在无可用索引的状态下启动。请稍后通过 '/manage/start-indexing' 构建新索引，或通过 '/manage/reload-index' 加载一个现有索引。")
            else:
                logger.info(f"默认索引 '{INDEX_NAME}' 加载成功。")
            return True
        except Exception as e:
            logger.error(f"初始化 FAISS ImageRetrieval 系统时发生严重错误: {e}", exc_info=True)
            retrieval_system_instance = None
            return False
    return True


def initialize_dino_processor():
    global dino_processor_instance
    if dino_processor_instance is None:
        logger.info("正在为 API 初始化 GroundingDINOProcessor...")
        try:
            if not os.path.exists(DINO_CONFIG_PATH) or not os.path.exists(DINO_MODEL_PATH):
                logger.error(f"GroundingDINO 配置文件 ({DINO_CONFIG_PATH}) 或模型文件 ({DINO_MODEL_PATH}) 未找到。")
                return False
            dino_processor_instance = GroundingDINOProcessor(
                config_path=DINO_CONFIG_PATH,
                model_path=DINO_MODEL_PATH,
                device=DEVICE
            )
            logger.info("GroundingDINOProcessor 初始化成功。")
            return True
        except Exception as e:
            logger.error(f"初始化 GroundingDINOProcessor 时发生错误: {e}", exc_info=True)
            dino_processor_instance = None
            return False
    return True

@app.on_event("startup")
async def startup_event():
    """FastAPI 应用启动时执行的事件处理函数，用于预加载模型。"""
    logger.info(f"API 服务启动中... 使用设备: {DEVICE}")
    faiss_ready = initialize_faiss_retrieval_system()
    dino_ready = initialize_dino_processor()
    
    if not faiss_ready:
        logger.error("FAISS 系统未能成功初始化")
    if not dino_ready:
        logger.error("GroundingDINO 系统未能成功初始化")
    if faiss_ready and dino_ready:
        logger.info("所有服务均已成功初始化")
        


async def _perform_search_for_image(image_bytes: bytes, dino_text_prompt: str, similarity_threshold: float, top_k: int):
    
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        query_image_numpy_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if query_image_numpy_bgr is None:
            raise ValueError("cv2.imdecode 返回 None，可能是无效的图像格式或文件已损坏。")
    except Exception as e:
        logger.error(f"解码图像时出错: {e}", exc_info=True)
        return {"status": "error", "message": f"无法解码图像文件: {e}"}

    logger.info(f"API 调用：使用DINO提取ROI，提示: '{dino_text_prompt}'")
    try:
        list_of_roi_pil_images = dino_processor_instance.get_rois_from_image(
            image_input=query_image_numpy_bgr, 
            text_prompt=dino_text_prompt
        )
    except Exception as e:
        logger.error(f"GroundingDINO ROI 提取过程中发生错误: {e}", exc_info=True)
        return {"status": "error", "message": f"ROI 提取失败: {e}"}

    
    encoded_rois = []
    if list_of_roi_pil_images:
        for roi_pil in list_of_roi_pil_images:
            buffered = io.BytesIO()
            roi_pil.save(buffered, format="PNG")
            encoded_rois.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    if not list_of_roi_pil_images:
        return {
            "status": "success",
            "message": f"未能从查询图片中基于提示 '{dino_text_prompt}' 提取到有效的ROI区域。未执行FAISS搜索。",
            "rois_extracted_count": 0,
            "extracted_rois_base64": [],
            "search_results": []
        }

    aggregated_results_dict = {}
    for i, roi_pil_img in enumerate(list_of_roi_pil_images):
        roi_idx_for_log = i + 1
        logger.info(f"API 调用：正在为 ROI #{roi_idx_for_log} 进行 FAISS 搜索...")
        try:
            search_output_for_roi = retrieval_system_instance.search_globally(
                query_image_input=roi_pil_img,
                k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            if search_output_for_roi.get("status") == "success" and search_output_for_roi.get("results"):
                for res_item in search_output_for_roi["results"]:
                    img_path = res_item['image_path']
                    current_similarity = res_item['similarity']
                    if img_path not in aggregated_results_dict or current_similarity > aggregated_results_dict[img_path]['similarity']:
                        aggregated_results_dict[img_path] = {
                            'similarity': current_similarity,
                            'image_path': img_path,
                            'source_roi_index': roi_idx_for_log
                        }
            else:
                return {
                    'status': 'error',
                    'message': '索引未加载、元数据不完整或未构建、或者索引为空,无法执行搜索。请检查索引文件。',
                    "rois_extracted_count": 0,
                    "extracted_rois_base64": [],
                    "search_results": []
                }
        except Exception as e:
            logger.error(f"为 ROI #{roi_idx_for_log} 进行 FAISS 搜索时发生错误: {e}", exc_info=True)
    
    if not aggregated_results_dict:
        return {
            "status": "success",
            "message": f"处理了 {len(list_of_roi_pil_images)} 个ROI，但未能找到满足相似度阈值 {similarity_threshold:.2f} 的图片。",
            "rois_extracted_count": len(list_of_roi_pil_images),
            "extracted_rois_base64": encoded_rois,
            "search_results": []
        }

    final_sorted_list = sorted(list(aggregated_results_dict.values()), key=lambda x: x['similarity'], reverse=True)
    final_top_k_output = final_sorted_list[:top_k]

    return {
        "status": "success",
        "message": f"搜索成功。处理了 {len(list_of_roi_pil_images)} 个ROI，找到 {len(final_top_k_output)} 个聚合后的相似结果。",
        "rois_extracted_count": len(list_of_roi_pil_images),
        "extracted_rois_base64": encoded_rois,
        "search_results": final_top_k_output
    }



@app.post("/search", summary="对单个图像进行相似性搜索")
async def search_images_api(
    query_image: UploadFile = File(..., description="要查询的图像文件。"),
    dino_text_prompt: str = Form(DEFAULT_DINO_TEXT_PROMPT, description="用于 GroundingDINO ROI 检测的文本提示。"),
    similarity_threshold: float = Form(0.65, ge=0.1, le=1.0),
    top_k: int = Form(10, ge=1, le=50)
):
    """
    接收上传的图像文件和参数，首先使用 GroundingDINO 提取 ROI，
    然后对每个 ROI 使用 FAISS 在指定目录中搜索相似图像，并返回聚合后的结果。
    """
    image_bytes = await query_image.read()
    if not image_bytes:
        logger.error("API 调用：未提供图像文件或文件为空。")
        raise HTTPException(status_code=400, detail="查询图像文件为空或未提供。")
    
    async with api_lock:
        if retrieval_system_instance is None:
            raise HTTPException(status_code=503, detail="索引服务尚未初始化，无法重载索引。")
        result = await _perform_search_for_image(image_bytes=image_bytes, dino_text_prompt=dino_text_prompt, similarity_threshold=similarity_threshold, top_k=top_k)
    
    # 根据统一的返回结构进行判断
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "未知的内部错误"))
    
    # 如果成功，返回 content 部分
    return JSONResponse(status_code=200, content=result)



@app.post("/batch-search", summary="对批量上传的图像进行相似性搜索")
async def search_images_batch_api(
    query_images: List[UploadFile] = File(..., description="要批量查询的图像文件列表。"),
    dino_text_prompt: str = Form(DEFAULT_DINO_TEXT_PROMPT),
    similarity_threshold: float = Form(0.65, ge=0.1, le=1.0),
    top_k: int = Form(10, ge=1, le=50)
):
    if not query_images:
        raise HTTPException(status_code=400, detail="未提供任何查询图像文件。")
        
    batch_results = []
    async with api_lock:
        if retrieval_system_instance is None:
            raise HTTPException(status_code=503, detail="索引服务尚未初始化，无法重载索引。")
        for image_file in query_images:
            logger.info(f"正在处理批量任务中的文件: {image_file.filename}")
            image_bytes = await image_file.read()
            
            response_payload = {}
            if not image_bytes:
                response_payload = {"status": "error", "message": "图像文件为空。"}
            else:
                result = await _perform_search_for_image(image_bytes=image_bytes, dino_text_prompt=dino_text_prompt, similarity_threshold=similarity_threshold, top_k=top_k)
                if result.get("status") == "error":
                    response_payload = {"status": "error", "message": result.get("message")}
                else:
                    response_payload = result
            
            response_payload["filename"] = image_file.filename
            batch_results.append(response_payload)
        
    return JSONResponse(status_code=200, content={"batch_results": batch_results})


INDEXING_STATUS = {
    "is_running": False,
    "start_time": None,
    "message": "未开始"
}

class IndexingRequest(BaseModel):
    url_list_txt: str
    index_name: str
    batch_size: int = 64
    download_concurrency: int = 100
    load_after: bool = True

# async def run_indexing_task(req: IndexingRequest):
#     """
#     后台线程中运行，调用索引构建
#     """
#     global INDEXING_STATUS, retrieval_system_instance
    
#     logger.info(f"后台索引任务启动: index_name='{req.index_name}', source='{req.url_list_txt}'")
#     INDEXING_STATUS["is_running"] = True
#     INDEXING_STATUS["start_time"] = datetime.now().isoformat()
#     INDEXING_STATUS["message"] = f"正在从 {req.url_list_txt} 构建索引 '{req.index_name}'..."

#     try:
#         if retrieval_system_instance:
#             await retrieval_system_instance.build_index_from_urls(
#                 url_list_txt=req.url_list_txt,
#                 index_base_name=req.index_name,
#                 index_dir=INDEX_DIR,
#                 batch_size=req.batch_size,
#                 download_concurrency=req.download_concurrency
#             )
#             INDEXING_STATUS["message"] = f"索引 '{req.index_name}' 构建成功完成！"
#             logger.info(f"后台索引任务 '{req.index_name}' 成功完成。")
            
#             if req.load_after:
#                 logger.info(f"根据请求，将自动加载新构建的索引: '{req.index_name}'")
                
#                 async with api_lock:
#                     logger.info("获取到锁，开始重载索引...")
#                     load_success = retrieval_system_instance.load_index(req.index_name, INDEX_DIR)
#                     if load_success:
#                         INDEXING_STATUS["message"] += f" 并已成功加载到内存。"
#                         logger.info(f"新索引 '{req.index_name}' 已成功加载。")
#                     else:
#                         INDEXING_STATUS["message"] += f" 但自动加载失败，请手动加载。"
#                         logger.error(f"自动加载新索引 '{req.index_name}' 失败。")
#             logger.info("索引重载操作完成，已释放锁。")
#         else:
#             raise RuntimeError("FAISS 检索系统实例未初始化。")
            
#     except Exception as e:
#         error_message = f"后台索引任务失败: {e}"
#         INDEXING_STATUS["message"] = error_message
#         logger.error(error_message, exc_info=True)
#     finally:
#         INDEXING_STATUS["is_running"] = False
#         INDEXING_STATUS["start_time"] = None
#         logger.info("后台索引任务执行完毕，状态已重置。")

class NpyIndexingRequest(BaseModel):
    npy_features_path: str
    metadata_path: str
    index_name: str
    load_after: bool = True

async def run_npy_indexing_task(req: NpyIndexingRequest):
    """
    这个异步函数将在后台运行，它负责调用同步的 build_index_from_npy 方法。
    """
    global INDEXING_STATUS, retrieval_system_instance
    
    logger.info(f"后台 NPY 索引任务启动: index_name='{req.index_name}', features='{req.npy_features_path}'")
    INDEXING_STATUS.update({
        "is_running": True,
        "start_time": datetime.now().isoformat(),
        "message": f"正在从 .npy 文件 '{os.path.basename(req.npy_features_path)}' 构建索引 '{req.index_name}'..."
    })

    try:
        if not retrieval_system_instance:
            raise RuntimeError("FAISS 检索系统实例未初始化。")

        await asyncio.to_thread(
            retrieval_system_instance.build_index_from_npy,
            npy_features_path=req.npy_features_path,
            metadata_path=req.metadata_path,
            index_base_name=req.index_name,
            index_dir=INDEX_DIR
        )
        
        
        success_message = f"索引 '{req.index_name}' (来自 {req.npy_features_path}) 构建成功完成！"
        INDEXING_STATUS["message"] = success_message
        logger.info(f"后台 NPY 索引任务 '{req.index_name}' 成功完成。")
        
        if req.load_after:
            logger.info(f"根据请求，将自动加载新构建的索引: '{req.index_name}'")
            async with api_lock:
                logger.info("获取到锁，开始重载索引...")
                load_success = await asyncio.to_thread(
                    retrieval_system_instance.load_index,
                    req.index_name, 
                    INDEX_DIR
                )
                if load_success:
                    INDEXING_STATUS["message"] += f" 并已成功加载到内存。"
                    logger.info(f"新索引 '{req.index_name}' 已成功加载。")
                else:
                    INDEXING_STATUS["message"] += f" 但自动加载失败，请手动加载。"
                    logger.error(f"自动加载新索引 '{req.index_name}' 失败。")
            logger.info("索引重载操作完成，已释放锁。")
            
    except Exception as e:
        error_message = f"后台 numpy 索引任务失败: {e}"
        INDEXING_STATUS["message"] = error_message
        logger.error(error_message, exc_info=True)
    finally:
        INDEXING_STATUS.update({"is_running": False, "start_time": None})
        logger.info("后台 numpy 索引任务执行完毕，状态已重置。")


@app.post("/manage/start-indexing", summary="从 .npy 文件启动后台索引构建任务")
async def start_npy_indexing_api(request: NpyIndexingRequest, background_tasks: BackgroundTasks):
    """
    接收 .npy 特征文件和 .pkl 元数据文件的路径，启动一个后台任务来构建索引。
    此接口会立即返回，不会等待构建完成。
    """
    global INDEXING_STATUS
    if INDEXING_STATUS["is_running"]:
        raise HTTPException(status_code=409, detail=f"已有索引任务在运行中: {INDEXING_STATUS['message']}")
    
    # 检查输入文件在服务器上是否存在
    if not os.path.exists(request.npy_features_path):
        raise HTTPException(status_code=404, detail=f"特征文件 .npy 未找到: {request.npy_features_path}")
    if not os.path.exists(request.metadata_path):
        raise HTTPException(status_code=404, detail=f"元数据文件 .pkl 未找到: {request.metadata_path}")

    # 将我们的异步包装函数添加到后台任务中
    background_tasks.add_task(run_npy_indexing_task, request)
    
    return JSONResponse(
        status_code=202, # 202 Accepted
        content={"message": "从 .npy 文件构建索引的任务已成功启动，正在后台运行。"}
    )

# @app.post("/manage/start-indexing", summary="从URL列表启动后台索引构建任务")
# async def start_indexing_api(request: IndexingRequest, background_tasks: BackgroundTasks):
#     global INDEXING_STATUS
#     if INDEXING_STATUS["is_running"]:
#         raise HTTPException(status_code=409, detail=f"索引任务已在运行中: {INDEXING_STATUS['message']}")
    
#     if not os.path.exists(request.url_list_txt):
#         raise HTTPException(status_code=404, detail=f"URL列表文件未找到: {request.url_list_txt}")

#     background_tasks.add_task(run_indexing_task, request)
    
#     return JSONResponse(
#         status_code=202, 
#         content={"message": "索引构建任务已成功启动，正在后台运行。请通过 /manage/indexing-status 接口查询进度。"}
#     )

@app.get("/manage/indexing-status", summary="查询后台索引构建任务的状态")
async def get_indexing_status_api():
    return JSONResponse(status_code=200, content=INDEXING_STATUS)


class ReloadIndexRequest(BaseModel):
    index_name: str
    
@app.post("/manage/reload-index", summary="重新加载指定的FAISS索引")
async def reload_index_api(request: ReloadIndexRequest):
    """
    加载一个新的FAISS索引。
    """
    global retrieval_system_instance
    if retrieval_system_instance is None:
        raise HTTPException(status_code=503, detail="索引服务尚未初始化，无法重载索引。")

    logger.info(f"收到重载索引请求: '{request.index_name}'")
    
    async with api_lock:
        logger.info("获取到锁，开始重载索引...")
        try:
            success = retrieval_system_instance.load_index(request.index_name, INDEX_DIR)
            if success:
                message = f"索引 '{request.index_name}' 已成功加载并生效。"
                logger.info(message)
                return JSONResponse(status_code=200, content={"message": message})
            else:
                message = f"加载索引 '{request.index_name}' 失败。请检查索引文件是否存在或已损坏。"
                logger.error(message)
                raise HTTPException(status_code=404, detail=message)
        except Exception as e:
            logger.error(f"重载索引时发生未知错误: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"重载索引时发生内部错误: {e}")

