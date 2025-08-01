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


def FromQiGetFile(prefix, limit=1000, delimiter=None,marker=None, localfile=None):
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
        downloads(i,ret.get('items'))
        if eof:
            break
        i+=1
        marker = ret.get('marker')
    
def downloads(i:int,RetItems:list):

    RetItems = tqdm(RetItems,total=len(RetItems),desc=f"下载第{i}个batch")
    for RetItem in RetItems:
        url = parent_url + RetItem.get('key')
        project = url.split("/")[-2]
        os.makedirs(parent_savedir + project,exist_ok=True)
        savefile = os.path.join(parent_savedir,project,url.split("/")[-1])
        if os.path.exists(savefile):
            print(f"{savefile}已存在")
            continue
        Down_url(url,savefile)


def Down_url(url,savefile):
    response = requests.get(url=url)
    if response.status_code==200:
        with open(savefile,'wb') as pic:
            pic.write(response.content)
            
            
def run(division,AIid,month,ds):
    sqlcon = sqlite3.connect("/home/fuxin/AiPatrol/road_depoly/data/Road_LD_DB.db")
    cursor = sqlcon.cursor()

    projects = []

    # listdays = FromQiGetdir(prefix='ai-patrol/YT20241035/2025/05/')
    for d in ds:
        day = f'ai-patrol/{AIid}/2025/{month}/{d}/'
        p = FromQiGetdir(prefix=day)
        if p == None:
            continue
        projects += p
    print(f"发现项目数量:{len(projects)} 个")
    for project in projects:
        profinish_code=0
        projectname = os.path.basename(project.rstrip('/'))
        proj =pd.read_sql("SELECT * FROM projects WHERE route_code=?",params=(projectname,),con=sqlcon)
        if  proj.empty:            
            FromQiGetFile(project)
            
        #开始推理
            profinish_code = start.main(projectname)
            if profinish_code==200:
                #更新已经预测完的的状态
                cursor.execute(f"""INSERT INTO record (project,sfpre,sfsend) VALUES (?,1,0)""",(projectname,))
                sqlcon.commit()
                shutil.rmtree(f"/home/fuxin/AiPatrol/Tempdata/{projectname}")
                print(f"删除{projectname} 项目文件")
        #计算并上报报文


        # if True:
        #     continue
        sfsend =pd.read_sql("SELECT * FROM record WHERE project=? and sfsend=0",params=(projectname,),con=sqlcon)
        if not sfsend.empty:
            A = uploadData.Getrequests(projectname)
            A.TraversalProject(division)

            # 上报报文
            resp_json = SendReport.sendReport2shms(AIid,projectname)
            if resp_json.get("code")==200:
                # if resp_json.get("data").get("status")== "INSERT":
                #     cursor.execute(f"""UPDATE record SET sfsend = 2 WHERE project =? """,(projectname,))
                # else:
                cursor.execute(f"""UPDATE record SET sfsend = 1 WHERE project =? """,(projectname,))
                sqlcon.commit()
            else:
                print(resp_json)

        
    sqlcon.close()
            




if __name__ == '__main__':
    # # 上传文件
    # updata('test.txt', 'test.txt')
    # # 下载文件
    AIid = "YT20241021"
    division = 430405
    month = "07"
    ds = ['29']
    run(division=division,AIid=AIid,month=month,ds=ds)