服务器组的连接与使用



Moon+Zerotier说明
现在我们采用Moon服务器加速大家的体验，需要大家在本地命令行进行简单的操作，云端我已经配置好了。

首先检查一下你的zerotier服务目录（不是文件目录）是否在，不然不可以使用下面的脚本

Zerotier的本地工作文件夹通常来说在
Windows: C:\ProgramData\ZeroTier\One
Mac: /Library/Application Support/ZeroTier/One (在 Terminal 中应为 /Library/Application\ Support/ZeroTier/One)
Linux: /var/lib/zerotier-one

其他配置见手册

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

nohup与其输出重定向
nohup command &                                    //该命令的一般形式
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
六、disown -h
disown -h % pid 把进程pid挂上忽略hup信号，该进程不会因为断开而断掉，相当于中途转nohup

七、kill
前台进程的终止：Ctrl+c


后台kill
ps aux | grep lung_imaging.py |  awk '{print $2}' | xargs kill -9

ps -ef|grep lung|grep -v grep|cut -c 9-15|xargs kill -15


关闭/开启图形界面
init 3
如果在关闭了图形界面后想临时打开可以使用指令"init 5"或者"startx"进行操作，这两个指令的区别在于"init 5"指令会重启系统但是"startx"不会，所以建议使用"startx"。


缓存清理
缓存是什么：为了提高文件系统性能，内核利用一部分物理内存分配出缓冲区，用于缓存系统操作和数据文件，当内核收到读写的请求时，内核先去缓存区找是否有请求的数据，有就直接返回，如果没有则通过驱动程序直接操作磁盘。

程序用完了如果缓存不自动清理的话，可以用这个人工清理掉：
echo 1 > /proc/sys/vm/drop_caches

proc文件节点，是用户与内核进行通信和数据交换的一个通道，sys/vm/drop_caches字面也好理解，系统的虚拟内存模块保留的一个drop caches的接口，想这个接口发指令就能drop caches，linux这个命名也是6啊，看名知意。

echo的参数：
0 – 不释放
1 – 释放页缓存
2 – 释放dentries和inodes
3 – 释放所有缓存
3还是慎用吧，另外释放内存之前最好sync一下，linux本身将内存中的缓存写回磁盘的时间是30s（印象是这个，如果自己没有调过内核参数的话），sync就是告诉系统，把缓存的东西该写磁盘的写磁盘，老子不等你那个30s了，老子要的现在就要。嗯，就是这样。




Linux显存占用无进程清理方法
在跑Caffe、TensorFlow、pytorch之类的需要CUDA的程序时，强行Kill掉进程后发现显存仍然占用

这时候可以使用如下命令查看到top或者ps中看不到的进程
fuser -v /dev/nvidia*

接着杀掉显示出的进程（有多个，如pid=12345，一个个删除）：
kill -9 12345
kill -9 12345m

批量清理显卡中残留进程：
sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh

清理指定GPU显卡中残留进程，如GPU 2：
sudo fuser -v /dev/nvidia2 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh





linux下删除文件夹及下面所有文件

使用rm -rf 目录名字 命令即可
-r 就是向下递归，不管有多少级目录，一并删除
-f 就是直接强行删除，不作任何提示的意思
 
rm 不带参数 只能删除文件


linux更新文件权限
chmod -x 去除执行能力


创建/移动/重命名/复制文件夹

创建aaaaaaaaa文件夹
mkdir aaaaaaaaa

移动p05-fu文件夹到aaaaaaaaa文件夹下
mv /root/user/p05-fu /root/user/aaaaaaaaa/p05-fu

mv命令既可以重命名，又可以移动文件或文件夹。
例子：将目录A重命名为B
mv A B
例子：将/a目录移动到/b下，并重命名为c
mv /a /b/c

复制
cp 命令使用 -r 参数可以将 packageA 下的所有文件拷贝到 packageB 中：
cp -r /home/packageA/* /home/cp/packageB/
将一个文件夹复制到另一个文件夹下，以下实例 packageA 文件会拷贝到 packageB 中：
cp -r /home/packageA /home/packageB
运行命令之后 packageB 文件夹下就有 packageA 文件夹了。


查看目录内容

du file_path
目录的容量(du)语法格式

[plain] du [-ahskm] 文件或者目录名

参数解释-a ： 列出所有的文件与目录容量，因为默认仅统计目录的容量而已
-h: 以人们较易读的容量格式呈现(G/M/K)显示，自动选择显示的单位大小
-s : 列出总量而已，而不列出每个个别的目录占用容量
-k ： 以KB为单位进行显示
-m : 以MB为单位进行显示常用命令参考 查看当前目录大小[plain] du -sh ./
查看某一指定目录大小[plain] du -sh directory_name 

在指定目录下显示10个占用空间最大(最小)的目录或文件
最大：[plain] du -sh * | sort -nr | head 
最小：[php] du -sh * | sort -n | head




统计个数

统计当前目录下文件的个数（不包括目录）
ls -l | grep "^-" | wc -l
统计当前目录下文件的个数（包括子目录）
ls -lR| grep "^-" | wc -l
查看某目录下文件夹(目录)的个数（包括子目录）
ls -lR | grep "^d" | wc -l

检查更新时间
stat filename/file_path
Access Time：简写为atime，表示文件的访问时间。当文件内容被访问时，更新这个时间 
Modify Time：简写为mtime，表示文件内容的修改时间，当文件的数据内容被修改时，更新这个时间。 
Change Time：简写为ctime，表示文件的状态时间，当文件的状态权限被修改时，更新这个时间，例如文件的链接数，大小，权限，Blocks数。


查看登陆记录

 last -f /var/log/wtmp
该日志文件永久记录每个用户登录、注销及系统的启动、停机的事件。因此随着系统正常运行时间的增加，该文件的大小也会越来越大

目前根据https://blog.csdn.net/jctian000/article/details/81773255
做了自动log的功能

Vim相关信息
参考资料https://www.runoob.com/linux/linux-vim.html

vim xxx.xx 对xxx.xx使用vim文本编辑

1.vi/vim 的基本使用方法
基本上 vi/vim 共分为三种模式，分别是命令模式（Command mode），输入模式（Insert mode）和底线命令模式（Last line mode）。 这三种模式的作用分别是：

命令模式：
用户刚刚启动 vi/vim，便进入了命令模式。
此状态下敲击键盘动作会被Vim识别为命令，而非输入字符。比如我们此时按下i，并不会输入一个字符，i被当作了一个命令。
以下是常用的几个命令：
* 		i 切换到输入模式，以输入字符。
* 		x 删除当前光标所在处的字符。
* 		: 切换到底线命令模式，以在最底一行输入命令。
若想要编辑文本：启动Vim，进入了命令模式，按下i，切换到输入模式。
命令模式只有一些最基本的命令，因此仍要依靠底线命令模式输入更多命令。

输入模式
在命令模式下按下i就进入了输入模式。
在输入模式中，可以使用以下按键：
* 		字符按键以及Shift组合，输入字符
* 		ENTER，回车键，换行
* 		BACK SPACE，退格键，删除光标前一个字符
* 		DEL，删除键，删除光标后一个字符
* 		方向键，在文本中移动光标
* 		HOME/END，移动光标到行首/行尾
* 		Page Up/Page Down，上/下翻页
* 		Insert，切换光标为输入/替换模式，光标将变成竖线/下划线
* 		ESC，退出输入模式，切换到命令模式

底线命令模式
在命令模式下按下:（英文冒号）就进入了底线命令模式。
底线命令模式可以输入单个或多个字符的命令，可用的命令非常多。
在底线命令模式中，基本的命令有（已经省略了冒号）：
* 		q 退出程序
* 		w 保存文件
*                 wq 储存后离开
按ESC键可随时退出底线命令模式。


2.乱码的问题
解决方法是在 ~/.vimrc 中添加如下的配置：
set encoding=utf8

3.常用组合键
[Ctrl] + [f]   屏幕『向下』移动一页
[Ctrl] + [b]   屏幕『向上』移动一页




压缩zip
将 /home/html/ 这个目录下所有文件和文件夹打包为当前目录下的 html.zip：
zip -q -r html.zip /home/html
如果在我们在 /home/html 目录下，可以执行以下命令：
zip -q -r html.zip *
从压缩文件 cp.zip 中删除文件 a.c
zip -dv cp.zip a.c

解压zip
解压当前目录的某个zip到当前目录，生成新路径/文件
unzip zipped_file.zip
解压当前目录的某个zip到新路径
unzip zipped_file.zip -d unzipped_directory




文件传输      服务器VPN下的ssh上传/下载文件

1、从本地传送文件到服务器

将 /home 目录中的 a.jsp 文件从本地传送到服务器 /home 目录下
scp /home/a.jsp root@xxx.xxx.xxx.xxx:/home

2、从服务器下载文件到本地

将服务器的 /home 目录中的a.jsp文件下载到本地的/home目录
scp root@xxx.xxx.xxx.xxx:/home/a.jsp /home

3、从本地传送目录到服务器

将本地的 /home 中的 local_dir 目录传送到服务器的 /home 目录
scp -r /home/local_dir root@xxx.xxx.xxx.xxx:/home

4、从服务器下载目录到本地

将服务器的 /home 目录中的 dir 目录下载到本地的 /home 目录
scp -r root@192.168.0.101:/home/dir /home



查看登录ip记录

Linux查看/var/log/wtmp文件查看IP登陆
last -f /var/log/wtmp


可能遇到的bug


pycharm 用远程环境时报错bash: line 0: cd: /home/tmp: No such file or directory
https://blog.csdn.net/zhuoyuezai/article/details/88121835

Pycharm:Can't get remote credentials for deployment server的解决办法
https://blog.csdn.net/qian2213762498/article/details/85634502
以及看一下是否有正确设置解释器，它可能没有识别成ssh解释器

Pycharm退出pytest模式（run pytest in模式）
https://blog.csdn.net/u011318077/article/details/88090830

RuntimeError: received 0 items of ancdata
https://www.jianshu.com/p/b67ab03cd7a1
https://blog.csdn.net/weixin_30419799/article/details/99989582

Linux部署zerotier局域网工具
https://blog.csdn.net/weixin_43944305/article/details/103107786

linux下开启SSH，并且允许root用户远程登录,允许无密码登录
https://www.cnblogs.com/exmyth/p/10403079.html

linux下的文件都变成了绿色
https://zhidao.baidu.com/question/752028904199993644.html?qbl=relate_question_1&word=linux%CE%C4%BC%FE%B1%E4%C2%CC%C9%AB

Linux 系统启动后出现(initramfs) 处理办法
https://blog.csdn.net/qq_29855509/article/details/105609721

Ubuntu命令卸载软件
https://blog.csdn.net/luckydog612/article/details/80877179

安装htop
https://www.cnblogs.com/humor-bin/p/12785597.html

Linux配置系统path，将anacoda的path设置到系统路径
https://www.cnblogs.com/youyoui/p/10680329.html

SWAP知识
https://blog.csdn.net/hzj_001/article/details/89387194
https://blog.csdn.net/wenwst/article/details/85065383
