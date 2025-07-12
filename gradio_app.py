import base64
import io
import os
from pathlib import Path
import PIL
import gradio as gr
import logging
import sys
import requests
import json
import cv2
import numpy as np
import urllib.parse


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv('API_BASE_URL', "http://localhost:8989")
API_SEARCH_ENDPOINT = f"{API_BASE_URL}/search"


DEFAULT_DINO_TEXT_PROMPT_UI = os.getenv('DEFAULT_DINO_TEXT_PROMPT_UI', "whole street view")


def gradio_search_via_api(query_image_numpy, target_dir_path, dino_prompt_text_ui,
                           similarity_thresh_ui, top_k_ui):
    
    if query_image_numpy is None:
        return "错误：请上传一张查询图片。", [], []
    if not target_dir_path or not target_dir_path.strip():
        return "错误：请输入目标目录路径。", [], []

    current_dino_prompt = dino_prompt_text_ui.strip() if dino_prompt_text_ui and dino_prompt_text_ui.strip() else DEFAULT_DINO_TEXT_PROMPT_UI
    
    try:
        similarity_thresh = float(similarity_thresh_ui)
        top_k = int(top_k_ui)
        if not (0.0 <= similarity_thresh <= 1.0):
            return "错误：相似度阈值应在 0.0 和 1.0 之间。", [], []
        if top_k <= 0:
            return "错误：返回结果数量必须大于0。", [], []
    except ValueError:
        return "错误：相似度阈值或返回结果数量的格式不正确。", [], []

    try:
        is_success, image_bytes_encoded = cv2.imencode(".jpg", query_image_numpy)
        if not is_success:
            raise ValueError("无法将 NumPy 图像编码为 JPEG 字节流。")
        image_file_bytes_tuple = ('query_image.jpg', image_bytes_encoded.tobytes(), 'image/jpeg')
    except Exception as e:
        logger.error(f"转换查询图片为字节流时出错: {e}", exc_info=True)
        return f"处理查询图片失败: {e}", [], []

    
    files_payload = {'query_image': image_file_bytes_tuple}
    data_payload = {
        'target_directory': target_dir_path,
        'dino_text_prompt': current_dino_prompt,
        'similarity_threshold': similarity_thresh,
        'top_k': top_k
    }

    logger.info(f"准备向 API 发送搜索请求: URL='{API_SEARCH_ENDPOINT}', DataKeys={list(data_payload.keys())}")

    try:
        response = requests.post(API_SEARCH_ENDPOINT, files=files_payload, data=data_payload, timeout=20)
        
        response.raise_for_status() 
        
        # 解析 JSON 响应
        api_response_data = response.json()
        logger.info(f"成功从 API 获取响应: {api_response_data.get('message', '无消息')}")

    except requests.exceptions.ConnectionError:
        error_msg = f"错误：无法连接到后端 API 服务 ({API_SEARCH_ENDPOINT})。请确保服务正在运行并且网络通畅。"
        logger.error(error_msg)
        return error_msg, [], []
    except requests.exceptions.Timeout:
        error_msg = f"错误：后端 API 请求超时 ({API_SEARCH_ENDPOINT})。服务可能处理时间过长或负载过高。"
        logger.error(error_msg)
        return error_msg, [], []
    except requests.exceptions.HTTPError as http_err:
        error_msg_detail = f"HTTP 错误 {http_err.response.status_code}。"
        try: # 尝试从 API 响应中获取更详细的错误信息
            error_detail_from_api = http_err.response.json().get("detail", http_err.response.text)
            error_msg_detail += f" API详情: {error_detail_from_api}"
        except json.JSONDecodeError: # 如果 API 返回的不是 JSON
            error_msg_detail += f" 原始响应: {http_err.response.text}"
        logger.error(f"API 返回 HTTP 错误: {error_msg_detail}")
        return f"错误：后端 API 返回错误。{error_msg_detail}", [], []
    except json.JSONDecodeError:
        error_msg = "错误：无法解码来自后端 API 的 JSON 响应。API 可能返回了非预期的格式。"
        logger.error(f"{error_msg} 响应文本: {response.text if 'response' in locals() else '无响应对象'}")
        return error_msg, [], []
    except Exception as e: # 其他未知错误
        error_msg = f"调用 API 时发生未知错误: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg, [], []

    status_message = api_response_data.get("message", "已从API接收到响应，但无具体消息。")
    search_results_from_api = api_response_data.get("search_results", [])
    rois_extracted_count = api_response_data.get("rois_extracted_count", 0)

    if rois_extracted_count > 0 and not search_results_from_api:
        if "未能找到满足相似度阈值" not in status_message:
            status_message += f" (提取到 {rois_extracted_count} 个ROI，但未找到满足条件的相似图片)"

    if not search_results_from_api:
        return status_message, [], [], []
    rois_base64_from_api = api_response_data.get("extracted_rois_base64", [])
    roi_gallery_output = []
    if rois_base64_from_api:
        logger.info(f"正在解码 {len(rois_base64_from_api)} 个ROI图片用于在Gradio中显示...")
        for i, b64_string in enumerate(rois_base64_from_api):
            try:
                image_bytes = base64.b64decode(b64_string)
                roi_image = PIL.Image.open(io.BytesIO(image_bytes))
                roi_gallery_output.append((roi_image, f"ROI #{i+1}"))
            except Exception as e:
                logger.error(f"解码或加载ROI Base64字符串 #{i+1} 时出错: {e}")

    dataframe_output = []
    gallery_output = []
    dataframe_output = []

    for rank_idx, res_item in enumerate(search_results_from_api):
        try:
            similarity_score = res_item['similarity']
            image_p = res_item['image_path']
            source_roi_display = f"ROI #{res_item.get('source_roi_index', '未知')}"
            gallery_output.append(image_p)
            dataframe_output.append([
                rank_idx + 1,
                f"{similarity_score:.4f}",
                image_p,
                source_roi_display
            ])
        except KeyError as ke:
            logger.warning(f"API 返回的结果项中缺少预期的键: {ke} - 项目内容: {res_item}")
        except Exception as e:
            logger.error(f"格式化来自 API 的结果项 '{res_item.get('image_path', '未知路径')}' 时出错: {e}", exc_info=True)

    if not gallery_output and rois_extracted_count > 0 :
        status_message = f"处理了 {rois_extracted_count} 个ROI，但聚合后未找到任何可展示的相似图片。"
    logger.info(f"gallery_output: {gallery_output}")
    return status_message, roi_gallery_output, gallery_output, dataframe_output


if __name__ == '__main__':
    logger.info("Gradio 应用客户端启动。此应用将通过 API 调用后端服务进行图像检索。")
    logger.info(f"请确保后端 FastAPI 服务正在运行，并且可以通过 '{API_SEARCH_ENDPOINT}' 访问。")


    gradio_app_interface = gr.Interface(
        fn=gradio_search_via_api, # 使用调用 API 的新函数
        inputs=[
            gr.Image(type="numpy", label="上传查询图片", sources=["upload", "clipboard"]),
            gr.Textbox(label="目标目录路径", placeholder="绝对路径, /xxx/yyy/zzz"),
            gr.Textbox(label="GroundingDINO 文本提示", placeholder=f"留空则使用默认: '{DEFAULT_DINO_TEXT_PROMPT_UI}'", value=""),
            gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.65, label="相似度阈值 (0.1-1.0)"),
            gr.Slider(minimum=1, maximum=50, step=1, value=10, label="返回结果数量 (Top K)")
        ],
        outputs=[
            gr.Textbox(label="状态信息"),
            gr.Gallery(label="提取出的ROI区域", show_label=True, elem_id="roi_gallery", columns=5, height="auto", object_fit="contain"),
            gr.Gallery(label="相似图片集", show_label=True, elem_id="gallery", columns=5, height="auto", object_fit="contain"),
            gr.Dataframe(
                headers=["排名", "相似度", "图片路径", "检测ROI来源"],
                label="详细结果",
                wrap=True,
                column_widths=["5%", "10%", "70%", "15%"]
            )
        ],
        title="图像检索客户端",
        allow_flagging="never",
    )

    logger.info("准备启动 Gradio 服务界面...")
    try:
        gradio_app_interface.launch(server_name="0.0.0.0", server_port=10508, share=False)
    except Exception as e:
        logger.critical(f"启动 Gradio 服务界面失败: {e}", exc_info=True)