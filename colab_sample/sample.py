"""
代码模板  版本：2021年9月21日 19：35

作者：张天翊

"""
import os

print('hello world')

# 注意文件路径
# 数据路径（在另一个盘data盘）'/data/zhanglab_headneck' + '项目名称'
data_root = os.path.join('/data/zhanglab_headneck', 'sample_data')
# 你的代码+材料文件路径 '/home/zhanglab_headneck/code/' + '你的文件夹' # 注意需要在系统切换路径
Path = os.path.join('/home/zhanglab_headneck/code', 'colab_sample')
# 用于记录tensorboard等等输出的文件夹
runs_path = os.path.join('/home/zhanglab_headneck/runs', 'colab_sample')

# sample代码段
sample_output_list = []
data_list = os.listdir(data_root)
for i in data_list:
    print(i)
    sample_output_list.append(i)

if os.path.exists(runs_path):
    pass
else:
    os.mkdir(runs_path)

for a_output_path in sample_output_list:  # 在runs_path输出一堆文件夹测试程序
    a_sample_output_path = os.path.join(runs_path, a_output_path)
    os.mkdir(a_sample_output_path)

print('hello colab-github！！！')

# 你的代码段
# 你的代码段
# 你的代码段


# 推荐学习使用notifyemail库来实现邮件通知与附件发送等工作:)
# pip install notifyemail
# import notifyemail as notify
