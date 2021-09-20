# Insight
大家一起学习 AI：计算机视觉CV领域

维护：
宋凡，张天翊



练习项目：
5种花朵的AI分类器


基础知识：教材
http://zh.d2l.ai


Tensorflow官网的流程，说的非常非常详细！
 大家看这个作为例子更好
 https://tensorflow.google.cn/tutorials/images/classification#create_a_dataset


 https://tensorflow.google.cn/tutorials/load_data/images
 
 
 
 Deep learning CNN材料


一。

我整理了几个材料，有些之前可能发给过大家，写的非常非常好

建议先看书的前面部分，看这个书先看到这一页http://zh.d2l.ai/chapter_prerequisite/install.html就可以。

1.cnn1
https://zhuanlan.zhihu.com/p/27908027
2.cnn2
https://zhuanlan.zhihu.com/p/28173972
3.更多数学的说明
https://zhuanlan.zhihu.com/p/27642620

周三 一同学习
https://www.jianshu.com/p/76efdbe4f47f

https://www.jianshu.com/p/40ef4ad8ad50


机器学习和深度学习中的正则化方法简介
https://mp.weixin.qq.com/s?__biz=MzUxNTY1MjMxNQ==&mid=2247484580&idx=1&sn=94e6c115efd24cfb1aad107fdf719383&chksm=f9b22b10cec5a2066cc1d646342a02755ad91089354bdf7a0a508069c58948338a09288c17a9&scene=21#wechat_redirect


二。

1.轻量级 CNN 架构设计
https://mp.weixin.qq.com/s/VRDCC5bfYHGbO9Qb70m95A

2.keras中Conv， SeparableConv2D, DepthwiseConv2D三种卷积过程浅谈
https://blog.csdn.net/shawroad88/article/details/95222082

Tensorflow笔记/材料

0.Numpy
https://mp.weixin.qq.com/s/tO_-YEhO8lV2La3z1qv20A


深度学习tensorflow2入门教程
图片分类任务
https://tensorflow.google.cn/tutorials/images/classification#configure_the_dataset_for_performance


注意这几个里面是倒叙排版的，就离谱

1.python & tensorflow & keras 总结（一）
https://zhuanlan.zhihu.com/p/153360838

2.python & tensorflow & keras 总结（二）
https://zhuanlan.zhihu.com/p/213949748

3.python & tensorflow & keras 总结（三）
https://zhuanlan.zhihu.com/p/260429105




进阶+新框架的一系列参考文献

1.泛化性乱弹：从随机噪声、梯度惩罚到虚拟对抗训练
https://kexue.fm/archives/7466

2.擦除：提升 CNN 特征可视化的 3 种重要手段
https://mp.weixin.qq.com/s/j5P2PcFRTdhddEPUPeEraA

3.《Attention is All You Need》浅读（简介+代码）
https://kexue.fm/archives/4765

4.突破瓶颈，打造更强大的Transformer（配合上一篇）
https://kexue.fm/archives/7325

5.再来一顿贺岁宴：从K-Means到Capsule
https://kexue.fm/archives/5112/comment-page-2#comments








LINUX的使用说明
cd转到
ls看有哪些文件/文件夹
pwd查看当前绝对路径
conda activate tf2
前台跑 python xxx.py
后台跑 nohup python xxx.py &

nohup会在代码路径生成一个nohup.out文件

查看文件（1000是行数，可以改大一点）
tail -1000 nohup.out

linux top命令VIRT,RES,SHR,DATA的含义

VIRT：virtual memory usage 虚拟内存 1、进程“需要的”虚拟内存大小，包括进程使用的库、代码、数据等
2、假如进程申请100m的内存，但实际只使用了10m，那么它会增长100m，而不是实际的使用量
RES：resident memory usage 常驻内存
1、进程当前使用的内存大小，但不包括swap out
2、包含其他进程的共享
3、如果申请100m的内存，实际使用10m，它只增长10m，与VIRT相反
4、关于库占用内存的情况，它只统计加载的库文件所占内存大小
SHR：shared memory 共享内存
1、除了自身进程的共享内存，也包括其他进程的共享内存
2、虽然进程只使用了几个共享库的函数，但它包含了整个共享库的大小
3、计算某个进程所占的物理内存大小公式：RES – SHR
4、swap out后，它将会降下来
DATA
1、数据占用的内存。如果top没有显示，按f键可以显示出来。
2、真正的该程序要求的数据空间，是真正在运行中要使用的。
top 运行中可以通过 top 的内部命令对进程的显示方式进行控制。内部命令如下：
s – 改变画面更新频率
l – 关闭或开启第一部分第一行 top 信息的表示
t – 关闭或开启第一部分第二行 Tasks 和第三行 Cpus 信息的表示
m – 关闭或开启第一部分第四行 Mem 和 第五行 Swap 信息的表示
N – 以 PID 的大小的顺序排列表示进程列表
P – 以 CPU 占用率大小的顺序排列进程列表
M – 以内存占用率大小的顺序排列进程列表
h – 显示帮助
n – 设置在进程列表所显示进程的数量
q – 退出 top
s – 改变画面更新周期
序号 列名 含义
a PID 进程id
b PPID 父进程id
c RUSER Real user name
d UID 进程所有者的用户id
e USER 进程所有者的用户名
f GROUP 进程所有者的组名
g TTY 启动进程的终端名。不是从终端启动的进程则显示为 ?
h PR 优先级
i NI nice值。负值表示高优先级，正值表示低优先级
j P 最后使用的CPU，仅在多CPU环境下有意义
k %CPU 上次更新到现在的CPU时间占用百分比
l TIME 进程使用的CPU时间总计，单位秒
m TIME+ 进程使用的CPU时间总计，单位1/100秒
n %MEM 进程使用的物理内存百分比
o VIRT 进程使用的虚拟内存总量，单位kb。VIRT=SWAP+RES
p SWAP 进程使用的虚拟内存中，被换出的大小，单位kb。
q RES 进程使用的、未被换出的物理内存大小，单位kb。RES=CODE+DATA
r CODE 可执行代码占用的物理内存大小，单位kb
s DATA 可执行代码以外的部分(数据段+栈)占用的物理内存大小，单位kb
t SHR 共享内存大小，单位kb
u nFLT 页面错误次数
v nDRT 最后一次写入到现在，被修改过的页面数。
w S 进程状态。（D=不可中断的睡眠状态，R=运行，S=睡眠，T=跟踪/停止，Z=僵尸进程）
x COMMAND 命令名/命令行
y WCHAN 若该进程在睡眠，则显示睡眠中的系统函数名
z Flags 任务标志，参考 sched.h
默认情况下仅显示比较重要的 PID、USER、PR、NI、VIRT、RES、SHR、S、%CPU、%MEM、TIME+、COMMAND 列。可以通过下面的快捷键来更改显示内容。
通过 f 键可以选择显示的内容。按 f 键之后会显示列的列表，按 a-z 即可显示或隐藏对应的列，最后按回车键确定。
按 o 键可以改变列的显示顺序。按小写的 a-z 可以将相应的列向右移动，而大写的 A-Z 可以将相应的列向左移动。最后按回车键确定。
按大写的 F 或 O 键，然后按 a-z 可以将进程按照相应的列进行排序。而大写的 R 键可以将当前的排序倒转。



Linux前台、后台、挂起、退出、查看命令汇总

command &  直接在后台运行程序
ctrl+c 退出前台的命令,不再执行
ctrl+z挂起前台命令暂停执行，回到shell命令行环境中
bg    将刚挂起的命令放到后台运行
bg %3  将第三个job放到后台运行
kill %3  杀死第三个job，不再执行
fg    将刚挂起的命令返回前台运行
fg %3  将第三个job返回前台运行
jobs   察看当前shell下运行的所有程序；带+表示最新的jobs；带-表示次新的jobs；其他jobs不带符号
nohup=no hang up，不挂断，如果你正在运行一个进程，而且你觉得在退出帐户时该进程还不会结束，那么可以使用nohup命令。该命令可以在你退出帐户/关闭终端之后继续运行相应的进程.长命令必须写在shell文件中，否则nohup不起作用
          nohup command &                 //该命令的一般形式
          nohup command > myout.file 2>&1 &      //log输出到myout.file，并将标准错误输出重定向到标准输出，再被重定向到myout.file


fg、bg、jobs、&、nohup、ctrl+z、ctrl+c 命令
一、&
加在一个命令的最后，可以把这个命令放到后台执行，如
watch  -n 10 sh  test.sh  &  #每10s在后台执行一次test.sh脚本
二、ctrl + z
可以将一个正在前台执行的命令放到后台，并且处于暂停状态。
三、jobs
查看当前有多少在后台运行的命令
jobs -l选项可显示所有任务的PID，jobs的状态可以是running, stopped, Terminated。但是如果任务被终止了（kill），shell 从当前的shell环境已知的列表中删除任务的进程标识。
四、fg
将后台中的命令调至前台继续运行。如果后台中有多个命令，可以用fg %jobnumber（是命令编号，不是进程号）将选中的命令调出。
￼
 五、bg
将一个在后台暂停的命令，变成在后台继续执行。如果后台中有多个命令，可以用bg %jobnumber将选中的命令调出。
六、kill
* 法子1：通过jobs命令查看job号（假设为num），然后执行kill %num
* 法子2：通过ps命令查看job的进程号（PID，假设为pid），然后执行kill pid
前台进程的终止：Ctrl+c

ps aux | grep lung_imaging.py |  awk '{print $2}' | xargs kill -9


linux下删除文件夹及下面所有文件

使用rm -rf 目录名字 命令即可
-r 就是向下递归，不管有多少级目录，一并删除
-f 就是直接强行删除，不作任何提示的意思
 
rm 不带参数 只能删除文件


查看目录大小
目录的容量(du)语法格式[plain] du [-ahskm] 文件或者目录名
参数解释-a ： 列出所有的文件与目录容量，因为默认仅统计目录的容量而已
-h: 以人们较易读的容量格式呈现(G/M/K)显示，自动选择显示的单位大小
-s : 列出总量而已，而不列出每个个别的目录占用容量
-k ： 以KB为单位进行显示
-m : 以MB为单位进行显示常用命令参考 查看当前目录大小[plain] du -sh ./
查看某一指定目录大小[plain] du -sh directory_name 在指定目录下显示10个占用空间最大(最小)的目录或文件最大：[plain] du -sh * | sort -nr | head 最小：[php] du -sh * | sort -n | head


