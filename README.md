## Gradio简易前端启动

```{bash}
python gradio_app.py
```

## 接口服务启动

```{bash}
sh ./scripts/start.sh
```

主服务日志地址: `./logs/main_server.log`


## 检索接口文档

#### 接口地址
`http://localhost:8989/search`

#### 请求方式
POST

#### 输入参数

| 参数名称           | 参数类型      | 是否必填 | 示例值                                                                 | 描述                                                                 |
|:-------------------|:--------------|:---------|:-----------------------------------------------------------------------|:---------------------------------------------------------------------|
| query_image        | UploadFile    | 是       | 上传的图像文件，如`xxx.jpg`                                        |要查询图像文件                 |
| target_directory   | str           | 是       | `/path/to/target/directory`                                           | 要搜索相似图像的目标文件夹的绝对路径                           |
| dino_text_prompt   | str           | 否       | 默认值为`DEFAULT_DINO_TEXT_PROMPT`，默认是`"whole street view"`                     | 用于 GroundingDINO ROI 检测的文本提示                              |
| similarity_threshold | float        | 否       | 默认值为`0.95`，范围为`0.1`到`1.0`                                    | 相似度阈值，筛选搜索结果中与查询图像相似度满足要求的图像 |
| top_k              | int           | 否       | 默认值为`10`，范围为`1`到`50`                                         | 返回的最相似结果数量。                                               |

#### 输出参数

| 参数名称                   | 参数类型      | 示例值                                                                 | 描述                                                                 |
|:---------------------------|:--------------|:-----------------------------------------------------------------------|:---------------------------------------------------------------------|
| message                    | str           | `"搜索成功。处理了 3 个ROI，找到 5 个聚合后的相似结果。"`               | 对搜索结果的总体描述，包括处理的 ROI 数量和找到的相似结果数量等信息 |
| rois_extracted_count       | int           | `3`                                                                    | 提取到的 ROI 区域数量。                                              |
| extracted_rois_base64      | list[str]     | `["base64_encoded_roi_image_1", "base64_encoded_roi_image_2", ...]`   | 提取到的 ROI 区域图像的 Base64 编码列表                            |
| search_results             | list[dict]    | `[{"similarity": 0.85, "image_path": "/path/to/image1.jpg", "source_roi_index": 1}, ...]` | 搜索到的相似图像结果列表，每个结果包含相似度、图像路径以及对应的 ROI 索引|

## 示例

### 请求示例
```http
POST /search
Content-Type: multipart/form-data

query_image: (上传的图像文件)
target_directory: /path/to/target/directory
dino_text_prompt: whole street view
similarity_threshold: 0.95
top_k: 10
```

### 响应示例
```json
{
    "message": "搜索成功。处理了 3 个ROI，找到 5 个聚合后的相似结果。",
    "rois_extracted_count": 3,
    "extracted_rois_base64": [
        "base64_encoded_roi_image_1",
        "base64_encoded_roi_image_2",
        "base64_encoded_roi_image_3"
    ],
    "search_results": [
        {
            "similarity": 0.99,
            "image_path": "/path/to/image1.jpg",
            "source_roi_index": 1
        },
        {
            "similarity": 0.99,
            "image_path": "/path/to/image2.jpg",
            "source_roi_index": 2
        },
        {
            "similarity": 0.99,
            "image_path": "/path/to/image3.jpg",
            "source_roi_index": 1
        },
        {
            "similarity": 0.99,
            "image_path": "/path/to/image4.jpg",
            "source_roi_index": 3
        },
        {
            "similarity": 0.99,
            "image_path": "/path/to/image5.jpg",
            "source_roi_index": 2
        }
    ]
}
```