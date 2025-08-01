# -*- coding: utf-8 -*-
# flake8: noqa

from qiniu import Auth, put_file, etag, BucketManager
import requests
import os
from tqdm import tqdm
import start
import uploadData  
import SendReport
import sqlite3
import shutil
import pandas as pd
from feature_extract import AsyncFeatureProcessor





# import qiniu.config
parent_url = "http://aipatrol-dev.oss.hnldjt.com/"
parent_savedir = "/home/fuxin/AiPatrol/Tempdata/"
#需要填写你的 Access Key 和 Secret Key
access_key = 'IAM-ase9LRDLnwokK1F180iXGCXjuMqLU5BWpfTXujXD'
secret_key = 'Hr0JtohNZ_0J1kZK4Mg8Vzh_d1VjTODkQRLEqPRiT-CT'

#构建鉴权对象
q = Auth(access_key, secret_key)

#要上传的空间
bucket_name = 'aipatrol-dev'







#上传函数
def updata(key, localfile):
    print('开始上传')
#上传后保存的文件名
    key = key

    #生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, key, 3600)

    #要上传文件的本地路径
    localfile = localfile

    ret, info = put_file(token, key, localfile, version='v2')
    print(info)
    assert ret['key'] == key
    assert ret['hash'] == etag(localfile)
    
    
#下载数据
def FromQiGetdir(prefix, limit=1000, delimiter='/',marker=None, localfile=None):
    bucket = BucketManager(q)
    # 前缀
    prefix = prefix
    # 列举条目
    limit = limit
    # 列举出除'/'的所有文件以及以'/'为分隔的所有前缀
    delimiter = delimiter
    i = 1
    l = []
    while True:
        ret, eof, info = bucket.list(bucket_name, prefix, marker, limit, delimiter)
        
        # if marker==None:
        #     break
        p = ret.get('commonPrefixes')
        if p!=None:
            l+=ret.get('commonPrefixes')
        assert len(ret.get('items')) is not None

        if eof:
            return l
        i+=1
        marker = ret.get('marker')


def FromQiGetFile(prefix, limit=1000, delimiter=None, marker=None, localfile=None, collect_files=None):
    """
    从七牛云获取文件
    collect_files: 如果提供，将收集下载的文件信息
    """
    print(f"{prefix}开始下载")
    bucket = BucketManager(q)
    # 前缀
    prefix = prefix
    # 列举条目
    limit = limit
    # 列举出除'/'的所有文件以及以'/'为分隔的所有前缀
    delimiter = delimiter
    i = 1
    while True:
        ret, eof, info = bucket.list(bucket_name, prefix, marker, limit, delimiter)
        
        assert len(ret.get('items')) is not None
        downloads(i, ret.get('items'), collect_files)
        if eof:
            break
        i += 1
        marker = ret.get('marker')
    
def downloads(i: int, RetItems: list, collect_files: List[Tuple[str, str]] = None):
    RetItems = tqdm(RetItems, total=len(RetItems), desc=f"下载第{i}个batch")
    for RetItem in RetItems:
        url = parent_url + RetItem.get('key')
        project = url.split("/")[-2]
        os.makedirs(parent_savedir + project, exist_ok=True)
        savefile = os.path.join(parent_savedir, project, url.split("/")[-1])
        
        if os.path.exists(savefile):
            print(f"{savefile}已存在")
        else:
            Down_url(url, savefile)
        
        if collect_files is not None:
            bucket_key = f"{bucket_name}/{RetItem.get('key')}"
            collect_files.append((savefile, bucket_key))


def Down_url(url,savefile):
    response = requests.get(url=url)
    if response.status_code==200:
        with open(savefile,'wb') as pic:
            pic.write(response.content)
            
            
def run(division, AIid, month, ds, 
        extract_features=False,  # 是否提取特征
        model_path=None,
        config_path=None,
        feature_save_path=None,
        metadata_save_path=None,
        batch_size=256):  # 批处理大小
    """
    extract_features用来控制是否提取特征
    """
    sqlcon = sqlite3.connect("/home/fuxin/AiPatrol/road_depoly/data/Road_LD_DB.db")
    cursor = sqlcon.cursor()

    feature_processor = None
    if extract_features:
        if not all([model_path, config_path, feature_save_path, metadata_save_path]):
            raise ValueError("特征提取需要提供所有相关路径参数")
        

        feature_processor = AsyncFeatureProcessor(
            model_path=model_path,
            config_path=config_path,
            feature_save_path=feature_save_path,
            metadata_save_path=metadata_save_path,
            batch_size=batch_size
        )
        feature_processor.start()

    projects = []

    # 获取项目列表
    for d in ds:
        day = f'ai-patrol/{AIid}/2025/{month}/{d}/'
        p = FromQiGetdir(prefix=day)
        if p == None:
            continue
        projects += p
    print(f"发现项目数量:{len(projects)} 个")
    
    for project in projects:
        profinish_code = 0
        projectname = os.path.basename(project.rstrip('/'))
        proj = pd.read_sql("SELECT * FROM projects WHERE route_code=?", params=(projectname,), con=sqlcon)
        
        if proj.empty:
            
            collected_files = [] if extract_features else None
            
            # 下载文件
            FromQiGetFile(project, collect_files=collected_files)
            
            # 处理特征提取
            if extract_features and collected_files:
                if use_async:
                    # 异步处理 - 添加到队列后继续
                    feature_processor.add_files(collected_files)
                    logging.info(f"已添加 {len(collected_files)} 个文件到特征提取队列")
                else:
                    # 同步处理 - 等待完成
                    logging.info(f"开始为项目 {projectname} 提取 {len(collected_files)} 个文件的特征")
                    feature_processor.process_files(collected_files)
            
            # 开始推理（异步模式下不会被阻塞）
            profinish_code = start.main(projectname)
            if profinish_code == 200:
                # 更新已经预测完的的状态
                cursor.execute(f"""INSERT INTO record (project,sfpre,sfsend) VALUES (?,1,0)""", (projectname,))
                sqlcon.commit()
                shutil.rmtree(f"/home/fuxin/AiPatrol/Tempdata/{projectname}")
                print(f"删除{projectname} 项目文件")
        
        # 后续处理保持不变...
    
    # 如果使用异步处理，等待所有任务完成
    if extract_features and use_async and feature_processor:
        feature_processor.stop()
    
    sqlcon.close()

if __name__ == '__main__':
    # 使用示例
    AIid = "YT20241021"
    division = 430405
    month = "07"
    ds = ['29']
    
    # GPU版本 - 使用异步处理，不阻塞主流程
    run(division=division, 
        AIid=AIid, 
        month=month, 
        ds=ds,
        extract_features=True,
        model_path="/path/to/your/model.pth",
        config_path="/path/to/your/config.json",
        feature_save_path="/path/to/features.npy",
        metadata_save_path="/path/to/metadata.pkl",
        batch_size=256 
    )