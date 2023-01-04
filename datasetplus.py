import json
import sys
import os, time, gc
import copy
# import synonyms
# 手动构建同类词词典（此处近义词为替换后语义通顺即可，比如“开始导航”可以替换为“重新导航”

navigation=["保持导航","继续导航","开始导航","恢复导航","导航开始","导航","重新导航","导"]
search=["查","查找","找","寻找","搜","重新搜索","搜索"]
shutdown=["停止","撤","结束","关掉","退掉","退出","关","停","闭","取消","关了","关闭"]
openup=["打开","开","恢复","开启"]
setup=["设","设置"]

Similar=[]
Similar.append(navigation)
Similar.append(search)
Similar.append(shutdown)
Similar.append(openup)
Similar.append(setup)


def load_dataset(data_path):
    datas = json.load(open(data_path, 'r',encoding='utf-8'))
    examples = []
    for data in datas:
        for utt in data:
            ex = utt
            examples.append(ex)
    return examples

train_path = os.path.join('./data', 'train.json')
train_dataset = load_dataset(train_path)
# newtmp=train_dataset[2]
# newtmp['manual_transcript']=newtmp['manual_transcript'].replace("导航","开始导航",1)
# print(newtmp)
total=[]
tmp=[]
for data in train_dataset:
    newtmp=copy.deepcopy(data)
    tmp.append(copy.deepcopy(newtmp))
    if len(data['semantic'])!=0:
        for seman in data['semantic']:
            if seman[1]=="操作":
                for similarlist in Similar:
                    if seman[2] in similarlist:
                        for word in similarlist:
                            if seman[2]!=word:
                                newtmp=copy.deepcopy(data)
                                newtmp['manual_transcript']=newtmp['manual_transcript'].replace(seman[2],word,1)
                                newtmp['asr_1best']=newtmp['asr_1best'].replace(seman[2],word,1)
                                for i in range(len(newtmp['semantic'])):
                                    if newtmp['semantic'][i]==seman:
                                        newtmp['semantic'][i][2]=word
                                        break  
                                tmp.append(copy.deepcopy(newtmp))
                        break

# print(tmp)
total.append(tmp)




json_str=json.dumps(total,indent=4,ensure_ascii=False)
with open('price.json','a',encoding='utf-8') as f:
    f.write(json_str)
# synlist=synonyms.nearby("高德地图",5)
# print(synlist[0])