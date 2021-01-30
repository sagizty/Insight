"""
版本：1。22。11：00
这个文件用来自动发送 输出log + 性能监控log + 追加的文件 到指定邮箱列表中
-------------------------------------------------------------------------------
2021.1.1 17:00  更新内容：本地日志文件保存在程序目录下的log文件夹内
2021.1.2 11:00  修复了计算均值写入日志间隔的bug
2021.1.22 11:00   返回值增加了服务器IP
-------------------------------------------------------------------------------
维护工作：
生成log部分/追加附件，张天翊
监控log部分，吕尚青
发送log部分，吴雨卓
注意这个函数的顺序很重要，不要改顺序!!!!!!
需要监控的程序可在启动时用如下代码调用本功能：
import os
import sys
# 将当前目录和父目录加入路径，使得文件可以调用本目录和父目录下的所有包和文件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general import notify
# 程序代码
...
notify.add_text("whatever u want to say")
notify.add_file(file_name） # 追加邮件附件的文件路径，可以是文件/文件夹（会自动zip），只需要在任意位置调用这个函数即可，可多次用
...
notify.send_log()  # 在自己代码的最后一段执行部分之后使用就行
-------------------------------------------------------------------------------
说明：
输出监控日志格式：
*****************LOG_Cache_2020_12_31_01_01*****************
内容
start time: 2020_12_31  01:01:14
end time: 2020_12_31  01:02:04
性能监控日志格式：
============================================
监控开始时间:    12月30日 23:04:12
采样间隔:  3  | 计算均值写入日志间隔:   1
============================================
时间: 12月30日 23:04:15   | CPU平均占用率: 0.79  | 内存占用率: 4.4
时间: 12月30日 23:04:16   | CPU平均占用率: 0.68  | 内存占用率: 4.4
============================================
监控结束时间:    12月30日 23:04:22
平均CPU占用率:   0.72  | 平均内存占用率:  4.4
最大CPU占用率:   0.74  | 最大内存占用率:  4.4
公邮：foe3305@163.com
密码：ddd888
如果想只发给自己，就把自己邮箱写进去：在自己代码的最后使用
notify.send_log(“1111@111.com”)
如果要发给多人，请传入一个包含多个str的元组/列表。
不写发给谁的话，默认会发给一个默认列表中的所有人，需要加入默认列表私聊zty。公邮在默认列表中。
"""

import smtplib
import zipfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# from email.header import Header
import sys
import datetime
import os
import time
import psutil
# import locale

import socket
import threading


def start_monitor(log_dir, log_name='server_status.log', report_time=300, sample_time=5):
    """
    启动函数
    :param log_dir:
    :param log_name:
    :param report_time:
    :param sample_time:
    :return:
    """
    log_name = alter_log_name(log_name)
    monitor_process = threading.Thread(target=server_monitor_process,
                                       args=(log_dir, log_name, report_time, sample_time))
    monitor_process.start()

    return monitor_process, log_name


def stop_monitor(monitor_process):
    global finish_process
    finish_process += 1
    monitor_process.join()


def server_monitor_process(log_dir, log_name='server_status.log', report_time=300, sample_time=5):
    """
    主函数
    :param log_dir: 日志保存目录
    :param log_name: 日志文件名
    :param report_time: 每次计算均值写入日志的间隔时间
    :param sample_time: 每次监控采样的间隔时间
    :return:
    """
    global finish_process
    next_time_to_report = report_time

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    write_information_to_log(log_dir, log_name, info_type='init', report_time=report_time, sample_time=sample_time)
    print('start monitoring')

    cpu_list = []
    mem_list = []

    cpu_avg_list = []
    mem_avg_list = []
    cpu_max_list = []
    mem_max_list = []
    while True:
        cpu_list.append(psutil.cpu_percent(interval=sample_time, percpu=True))
        mem_list.append(psutil.virtual_memory().percent)
        next_time_to_report -= sample_time

        # 进程结束后保存并退出
        if finish_process == 1:
            cpu_avg_list.append(calc_avg_cpu_usage_percentage(cpu_list))
            mem_avg_list.append(calc_avg_mem_usage_percentage(mem_list))
            cpu_max_list.append(calc_max_cpu_usage(cpu_list))
            mem_max_list.append(max(mem_list))
            save_server_log(cpu_avg_list[-1], mem_avg_list[-1], log_dir, log_name)

            cpu_avg = calc_avg_mem_usage_percentage(cpu_avg_list)
            mem_avg = calc_avg_mem_usage_percentage(mem_avg_list)
            cpu_max = calc_avg_mem_usage_percentage(cpu_max_list)
            mem_max = calc_avg_mem_usage_percentage(mem_max_list)
            write_information_to_log(log_dir, log_name, info_type='finish',
                                     cpu_avg=cpu_avg, mem_avg=mem_avg, cpu_max=cpu_max, mem_max=mem_max)
            return 0

        # 正常保存
        elif next_time_to_report <= 0:
            cpu_avg_list.append(calc_avg_cpu_usage_percentage(cpu_list))
            mem_avg_list.append(calc_avg_mem_usage_percentage(mem_list))
            cpu_max_list.append(calc_max_cpu_usage(cpu_list))
            mem_max_list.append(max(mem_list))
            save_server_log(cpu_avg_list[-1], mem_avg_list[-1], log_dir, log_name)
            cpu_list = []
            mem_list = []
            next_time_to_report = report_time


def calc_avg_cpu_usage_percentage(cpu_usage_list_divided_by_time):
    avg_cpu_usage = 0
    cnt = 0
    for _sample in cpu_usage_list_divided_by_time:
        for single_cpu_percentage in _sample:
            avg_cpu_usage += single_cpu_percentage
            cnt += 1
    avg_cpu_usage = avg_cpu_usage / cnt
    return round(avg_cpu_usage, 2)


def calc_max_cpu_usage(cpu_usage_list_divided_by_time):
    max_cpu_usage = 0
    for _sample in cpu_usage_list_divided_by_time:
        avg_usage = 0
        for single_cpu_percentage in _sample:
            avg_usage += single_cpu_percentage
        avg_usage /= len(_sample)
        if avg_usage > max_cpu_usage:
            max_cpu_usage = avg_usage
    return round(max_cpu_usage, 2)


def calc_avg_mem_usage_percentage(mem_usage_list_divided_by_time):
    avg_mem_usage = 0
    for _sample in mem_usage_list_divided_by_time:
        avg_mem_usage += _sample
    avg_mem_usage = avg_mem_usage / len(mem_usage_list_divided_by_time)
    return round(avg_mem_usage, 2)


def save_server_log(cpu_usage, mem_usage, log_dir, log_name):
    now_time = time.strftime('%m月%d日 %H:%M:%S', time.localtime(time.time()))
    format_save = r'时间: %s   | CPU平均占用率: %s  | 内存占用率: %s  ' % \
                  (now_time, str(cpu_usage), str(mem_usage))
    with open(os.path.join(log_dir, log_name), mode="a", encoding="utf-8") as f:
        f.write(format_save + '\n')
        f.close()


def write_information_to_log(log_dir, log_name, info_type, report_time=60, sample_time=5,
                             cpu_avg='', mem_avg='', cpu_max='', mem_max=''):
    """
    在日志的前后部写入统计信息
    :param log_dir:
    :param log_name:
    :param info_type:
    :param report_time:
    :param sample_time:
    :param cpu_avg:
    :param mem_avg:
    :param cpu_max:
    :param mem_max:
    :return:
    """
    current_time = time.strftime('%m月%d日 %H:%M:%S', time.localtime(time.time()))
    if info_type == 'init':
        status_statement = '============================================\n' \
                           '监控开始时间:    %s\n' \
                           '采样间隔:  %s  | 计算均值写入日志间隔:   %s  \n' \
                           '============================================\n' % \
                           (current_time, str(sample_time), str(report_time))
    elif info_type == 'finish':
        status_statement = '============================================\n' \
                           '监控结束时间:    %s\n' \
                           '平均CPU占用率:   %s  | 平均内存占用率:  %s\n' \
                           '最大CPU占用率:   %s  | 最大内存占用率:  %s\n' % \
                           (current_time, str(cpu_avg), str(mem_avg), str(cpu_max), str(mem_max))
    else:
        return 1
    with open(os.path.join(log_dir, log_name), mode="a", encoding="utf-8") as f:
        f.write(status_statement + '\n')
        f.close()


def alter_log_name(log_name):
    new_name = log_name
    suffix = ''
    if log_name[-4:] == '.log':
        new_name = log_name[:-4]
        suffix = '.log'
    new_name += time.strftime('_%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + suffix
    return new_name


def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹到指定路径
    :param dirpath: 目标文件夹路径:1212/12/c
    :param outFullName:  'aaa/bbb/c.zip'
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, 'w', zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标和路径，只对目标文件夹下边的文件及文件夹进行压缩（包括父文件夹本身）
        this_path = os.path.abspath('.')
        fpath = path.replace(this_path, '')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


def make_print_save_to_file(path='./'):
    """
    重写print,使得之后所有print内容都会被保存到log中， 默认调用时就要执行
    log名字为: Date 年_月_日.log
    :param path: 放log的路径
    :return:
    """

    class Logger(object):
        def __init__(self, processing_log_name="LOG_Default.log", path="./"):
            self.ori_stdout = sys.stdout
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, processing_log_name), "a", encoding='utf8', )
            self.start_time = time.time()
            self.text_content = []
            self.additional_file_list = []

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

        def close_log_and_put_back(self):
            self.log.close()
            sys.stdout = self.ori_stdout  # ori_stdout是原本的sys.stdout， 现在用来恢复

        def add_a_text(self, text_input):
            self.text_content.append(text_input)

        def add_a_file(self, file_name):
            self.additional_file_list.append(file_name)

    fileName = datetime.datetime.now().strftime('LOG_Cache_' + '%Y_%m_%d_%H_%M')
    sys.stdout = Logger(fileName + '.log', path=path)

    log_monitor = sys.stdout  # 现在的被重写的sys.stdout对象

    log_name = os.path.join(path, fileName + '.log')

    # 这里输出之后的所有的其他代码中的输出的print 内容即将自动备份写入到日志Date 年_月_日.log里
    print(fileName.center(60, '*'))  # 首先写个日志头

    return log_name, log_monitor


# 初始化，配置参数：

# 日志格式
log_type = ".rtf"

# 配置邮箱信息
mail_host = 'smtp.exmail.qq.com'
mail_user = 'notice@visionwyz.com'
mail_pass = '3cvPbaNucRHvNiJb'  # 腾讯企业邮箱的授权码
sender = mail_user
defaut_receivers = ('foe3305@163.com', '476017732@qq.com', 'wuyuzhuo@visionwyz.com')

place_to_save_log = os.path.join(os.getcwd(), 'log')
if not os.path.exists(place_to_save_log):
    os.mkdir(place_to_save_log)

log_cache_name, log_monitor = make_print_save_to_file(place_to_save_log)  # 重写print, 默认调用时就要执行

# 初始化monitor 监控log部分
lock = threading.Lock()
# locale.setlocale(locale.LC_CTYPE, 'chinese')  # 使用windows时有可能需要该代码
finish_process = 0

# 定义监控参数
server_log_dir = place_to_save_log  # 服务器server_log路径
server_log_name = 'server_status.log'  # 服务器server_log名字
report_time = 300  # 每次计算均值写入日志的间隔时间 300s
sample_time = 5  # 每次监控采样的间隔时间 5s

# 开始监控
monitor_process, server_log_name = start_monitor(server_log_dir, server_log_name, report_time, sample_time)


def add_text(text_input, log_monitor=log_monitor):
    """
    设置文件内容
    :param text_input: 追加的文件内容
    :param log_monitor: 系统对象（不需要管）
    :return:
    """
    log_monitor.add_a_text(text_input=text_input)


def add_file(file_name, log_monitor=log_monitor):
    """
    追加邮件附件，可以是文件/文件夹（会自动zip），只需要调用这个函数即可
    :param file_name: 追加的附件路径，可以是文件/文件夹（会自动zip）
    :param log_monitor: 系统对象（不需要管）
    :return:
    """
    log_monitor.add_a_file(file_name=file_name)
    print(file_name, " has been added to the mail attachment list as an additional file")


def get_host_ip():
    '''
    获取本机ip
    :return:
    '''
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def send_log(mail_list=defaut_receivers, log_cache_name=log_cache_name, log_monitor=log_monitor,
             server_log_dir=server_log_dir, server_log_name=server_log_name, log_type=log_type):
    """
    :param mail_list: 发送file_name到mail_list中的收件人
    mail_list可以是列表['xxx@xx.com','xxx@xx.com']或字符串'xxx@xx.com'
    :param log_cache_name: Log的Cache文件，如果报错那么就只有这个记录，不报错会存为正式的log
    :param log_monitor: 重写的sys.stdout对象
    :param log_type: 日志格式
    :param server_log_dir: 服务器server_log路径
    :param server_log_name: 服务器server_log名字
    :return:
    """
    # 此时结束监控
    stop_monitor(monitor_process)

    # 确定时间
    time_start = log_monitor.start_time
    time_end = time.time()

    print("\nProcessing finished !")
    print("start time:", time.strftime('%Y_%m_%d  %H:%M:%S', time.localtime(time_start)))
    print("end time:", time.strftime('%Y_%m_%d  %H:%M:%S', time.localtime(time_end)))
    print("source:", get_host_ip())

    # 定log文件名
    call_func_name = sys._getframe(1).f_code.co_filename.split('/')[-1]
    processing_log_name = call_func_name + '__' + time.strftime('%Y_%m_%d %H_%M_%S',
                                                                time.localtime(time_start)) + '_log'

    print("\nPreparing the email with auto log file :\n", processing_log_name, '\n', server_log_name, '\nas ', log_type)

    # 定发送对象
    if type(mail_list) == str:
        mail_list = [mail_list]
    else:
        mail_list = list(mail_list)

    # 确定邮件题目
    mail_title = '[LOG] ' + processing_log_name

    # 组织email内容
    message = MIMEMultipart()
    message['Subject'] = mail_title
    message['From'] = mail_user

    # 如果是收件人列表，做编码处理
    if len(mail_list) > 1:
        message['To'] = ";".join(mail_list)
    elif len(mail_list) == 1:  # 如果只是一个邮箱， 就发到这个邮箱
        message['To'] = mail_list[0]
    else:
        print("mail_list problem occur!")
        return -1

    # 处理文本
    mail_content = log_monitor.text_content
    if len(mail_content) > 0:
        text_item = 'Note:\n'
        for i in mail_content:
            text_item += str(i)
            text_item += '\n'
        message.attach(MIMEText(text_item, 'plain', 'utf-8'))

    # 处理追加附件
    additional_file_list = log_monitor.additional_file_list

    if len(additional_file_list) > 0:
        print("\nThe email attaching with additional files :")
        for i in additional_file_list:

            file_name = str(i).split('/')[-1]

            if os.path.isdir(i):  # 是文件夹, 要压缩
                file_loc = './' + file_name + '.zip'
                zipDir(i, file_loc)
                file_name = file_name + '.zip'

            else:  # 是文件
                file_loc = i

            with open(file_loc, 'rb') as Af:
                file = Af.read()
            try:
                Af.close()
                # 添加附件
                log_part = MIMEText(file, 'base64', 'utf-8')
                log_part["Content-Type"] = 'application/octet-stream'
                # 这里的filename可以任意写，写什么名字，邮件中显示什么名字
                log_part["Content-Disposition"] = 'attachment; filename="%s"' % file_name
                message.attach(log_part)
            except:
                print("Erro occur in adding additional file:", file_name)
            else:
                print("An additional file has been added to the mail:", file_name)
        print('\n')

    # 阻断log生成
    log_monitor.close_log_and_put_back()  # log close，于是可以读到log; 恢复还原原本的sys.stdout，同时阻断log生成.

    # 调取processing_log
    try:
        # 读入log文件(作为普通文本附件读入)
        with open(log_cache_name, 'r') as l:
            processing_log = l.read()
            l.close()
        if processing_log[0] is not '*':
            print("processing log title erro")
        else:
            os.rename(log_cache_name, os.path.join(place_to_save_log, processing_log_name + '.log'))
    except:
        print("processing log status erro")
        return -1
    else:
        print("processing log catched")

    # 调取server_log
    try:
        with open(os.path.join(server_log_dir, server_log_name), 'r') as f:
            server_log = f.read()
            f.close()
        if server_log[0] is not '=':
            print("server log title erro")
    except:
        print("server log status erro")
        return -1
    else:
        print("server log catched")

    # 附件1：processing_log
    log_part = MIMEText(processing_log, 'base64', 'utf-8')
    log_part["Content-Type"] = 'application/octet-stream'
    file = processing_log_name + log_type
    log_part[
        "Content-Disposition"] = 'attachment; filename="%s"' % file  # 这里的filename可以任意写，写什么名字，邮件中显示什么名字
    message.attach(log_part)

    # 附件2：server_log
    log_part = MIMEText(server_log, 'base64', 'utf-8')
    log_part["Content-Type"] = 'application/octet-stream'
    file = server_log_name + log_type
    log_part[
        "Content-Disposition"] = 'attachment; filename="%s"' % file  # 这里的filename可以任意写，写什么名字，邮件中显示什么名字
    message.attach(log_part)

    # 实例化，也是登录的过程
    smtp = smtplib.SMTP_SSL(mail_host)
    smtp.ehlo(mail_host)
    smtp.login(mail_user, mail_pass)
    smtp.sendmail(mail_user, mail_list, message.as_string())
    smtp.quit()
    print('发送log邮件成功，title: ', mail_title)
    print('如果没有，看看垃圾箱:)')