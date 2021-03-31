import pickle
import json
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

'''
json格式
[
  {'target_bbox': [0.39648438, 0.3408203, 0.07421875, 0.08007812],  # GT bbox
   'pred_bbox': [0.39571705, 0.35183546, 0.07052322, 0.10342153],  # 预测bbox
   'target_label': 0,  # 真实label
   'pred_label': 0,  # 预测label
   'imageid': 12345},  # 测试对象的id，这个值现在的pickle里面好像没有

   {'target_bbox': [0.39648438, 0.3408203, 0.07421875, 0.08007812],
   'pred_bbox': [0.39571705, 0.35183546, 0.07052322, 0.10342153],
   'target_label': 0,
   'pred_label': 0,
   'imageid': 12345},

   {'target_bbox': [0.39648438, 0.3408203, 0.07421875, 0.08007812],
   'pred_bbox': [0.39571705, 0.35183546, 0.07052322, 0.10342153],
   'target_label': 0,
   'pred_label': 0,
   'imageid': 12345},

   ]

'''


def detr_output_json_logger(outputs, targets):
    """
    outputs = model(samples)  # 一波数据输入并预测出来
    :param outputs: 一波数据的模型输出
    :param targets: 一波数据的信息
    :return:
    """
    log_json = []

    for item_idx in range(len(outputs)):
        json_file = {}

        item_pred = outputs[item_idx]
        logits = torch.argmax(item_pred['pred_logits'][0], axis=1)

        json_file['pred_boxes'] = item_pred['pred_boxes'][0][
            logits != 5].cpu()  # 请问这个【0】是为什么！是否需要！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        json_file['pred_cls'] = logits[logits != 5].cpu().numpy()

        # plotit(samples.tensors[0].cpu(), p_boxes, names, "./vis/vis_output/"+str(index)+".jpg")
        json_file['image_id'] = targets[item_idx][0]['image_id']

        json_file["target_bbox"] = targets[0]['boxes'].cpu().numpy()
        json_file["target_label"] = targets[item_idx][0]['labels'].cpu().numpy()

        log_json.append(json_file)

    # 每一波数据建立 json log
    return log_json


json_file = {}
y_true = []     # 真实标签，[0,1,2,3]
y_pred = []     # 预测标签，[0,1,2,3]
sns.set()
_, ax = plt.subplots()

# 遍历目录，获取读取所有pickle格式数据
for i in os.listdir('/opt/ldy/detr_2021/vis/pickle/'):
    with open("/opt/ldy/detr_2021/vis/pickle/%s" % str(i), "rb") as files:
        # 读取pickle数据
        file_dict = pickle.load(files)

        # pickle数据中的数组是np.array格式，需要转换成list
        json_file[i] = file_dict
        for i in file_dict:
            file_dict[i] = file_dict[i].tolist()

        # 部分pickle数据没有预测值，这里进行简单的判断筛选
        if file_dict['pred_label'] != [] and file_dict['pred_label'] != []:
            label_true = file_dict['target_label'][0]
            label_pred = file_dict['pred_label'][0]

            if label_pred in [0, 1, 2, 3] and label_true in [0, 1, 2, 3]:
                y_true.append(label_true)
                y_pred.append(label_pred)

# 绘制并打印混淆矩阵
C2 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
print(C2)


# 画图前的数据格式整理，将数据转化为带有坐标轴的dataframe格式
tick = ['A', 'B', 'E', 'G']
data = {}
for i in range(4):
    data[tick[i]] = C2[i]
pd_data = pd.DataFrame(data, index=tick, columns=tick)
print(pd_data)

# 绘制热力图，“fmt='.20g'”表示不采用科学计数法
sns.heatmap(pd_data, annot=True, fmt='.20g', ax=ax)  # 画热力图
ax.set_title('confusion matrix')  # 标题
ax.set_xlabel('predict')  # x轴
ax.set_ylabel('true')  # y轴
plt.show()


# 保存json
with open('/home/NSCLC-project/NSCLC_go/train_result.json', 'w') as f:
    json.dump(json_file, f)

