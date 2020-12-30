"""
版本 2020.12.30 01:00
这个文件用来自动发送log文件到指定邮箱

维护工作：
生成log部分，张天翊
监控log部分，吕尚青
发送log部分，吴雨卓

注意这个函数的顺序很重要，不要改顺序!!!!!!


需要监控的程序可在启动时用如下代码调用本功能：
-------------------------------------------------------------------------------
import notify

# 程序代码
...

notify.send_log() #建议在自己代码的最后一段执行部分之后使用就行

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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# from email.header import Header
import sys
import datetime
import os
import time
import psutil
# import locale

import threading


def start_monitor(log_dir, log_name='server_status.log', report_time=300, sample_time=3):
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


def server_monitor_process(log_dir, log_name='server_status.log', report_time=300, sample_time=3):
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
            save_log(cpu_avg_list[-1], mem_avg_list[-1], log_dir, log_name)

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
            save_log(cpu_avg_list[-1], mem_avg_list[-1], log_dir, log_name)
            cpu_list = []
            mem_list = []


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


def save_log(cpu_usage, mem_usage, log_dir, log_name):
    now_time = time.strftime('%m月%d日 %H:%M:%S', time.localtime(time.time()))
    format_save = r'时间: %s   | CPU平均占用率: %s  | 内存占用率: %s  ' % \
                  (now_time, str(cpu_usage), str(mem_usage))
    with open(os.path.join(log_dir, log_name), mode="a", encoding="utf-8") as f:
        f.write(format_save + '\n')
        f.close()


def write_information_to_log(log_dir, log_name, info_type, report_time=0, sample_time=0,
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
                           (current_time, str(report_time), str(sample_time))
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


def make_print_save_to_file(path='./'):
    """
    重写print,使得之后所有print内容都会被保存到log中， 默认调用时就要执行
    log名字为: Date 年_月_日.log
    :param path: 放log的路径
    :return:
    """

    class Logger(object):
        def __init__(self, filename="LOG_Default.log", path="./"):
            self.ori_stdout = sys.stdout
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )
            self.start_time = time.time()

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

        def close_log_and_put_back(self):
            self.log.close()
            sys.stdout = self.ori_stdout  # ori_stdout是原本的sys.stdout， 现在用来恢复

    fileName = datetime.datetime.now().strftime('LOG_Cache_' + '%Y_%m_%d_%H_%M')
    sys.stdout = Logger(fileName + '.log', path=path)

    current_stdout = sys.stdout  # 现在的被重写的sys.stdout对象

    log_name = path + fileName + '.log'

    # 这里输出之后的所有的其他代码中的输出的print 内容即将自动备份写入到日志Date 年_月_日.log里
    print(fileName.center(60, '*'))  # 首先写个日志头

    return log_name, current_stdout


# 初始化，配置参数：

# 日志格式
log_type = ".rtf"

# 配置邮箱信息
mail_host = 'smtp.exmail.qq.com'
mail_user = 'notice@visionwyz.com'
mail_pass = '3cvPbaNucRHvNiJb'  # 腾讯企业邮箱的授权码
sender = mail_user
defaut_receivers = ('foe3305@163.com', '476017732@qq.com', 'wuyuzhuo@visionwyz.com')

log_cache_name, current_stdout = make_print_save_to_file()  # 重写print, 默认调用时就要执行

# 初始化monitor 监控log部分
lock = threading.Lock()
# locale.setlocale(locale.LC_CTYPE, 'chinese')  # 使用windows时有可能需要该代码
finish_process = 0

# 定义监控参数
server_log_dir = '.'  # 服务器server_log路径
server_log_name = 'server_status.log'  # 服务器server_log名字
report_time = 300  # 每次计算均值写入日志的间隔时间 300s
sample_time = 5  # 每次监控采样的间隔时间 5s

# 开始监控
monitor_process, server_log_name = start_monitor(server_log_dir, server_log_name, report_time, sample_time)


def send_log(mail_list=defaut_receivers, log_cache_name=log_cache_name, current_stdout=current_stdout,
             server_log_dir=server_log_dir, server_log_name=server_log_name, log_type=log_type):
    """
    :param mail_list: 发送file_name到mail_list中的收件人
    mail_list可以是列表['xxx@xx.com','xxx@xx.com']或字符串'xxx@xx.com'
    :param log_cache_name: Log的Cache文件，如果报错那么就只有这个记录，不报错会存为正式的log
    :param current_stdout: 重写的sys.stdout对象
    :param log_type: 日志格式
    :param server_log_dir: 服务器server_log路径
    :param server_log_name: 服务器server_log名字

    :return:
    """
    # 此时结束监控
    stop_monitor(monitor_process)

    # 确定时间
    time_start = current_stdout.start_time
    time_end = time.time()

    print("\nstart time:", time.strftime('%Y_%m_%d  %H:%M:%S', time.localtime(time_start)))
    print("end time:", time.strftime('%Y_%m_%d  %H:%M:%S', time.localtime(time_end)))

    # 定log文件名
    call_func_name = sys._getframe(1).f_code.co_filename.split('/')[-1]
    file_name = call_func_name + '__' + time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time_start)) + '_log'

    print("sending the email attaching with file：\n", file_name, '\n', server_log_name, '\nas ', log_type)
    current_stdout.close_log_and_put_back()  # log close，于是可以读到log; 恢复还原原本的sys.stdout，同时阻断log生成.

    # 定发送对象
    if type(mail_list) == str:
        mail_list = [mail_list]
    else:
        mail_list = list(mail_list)

    # 调取processing_log
    try:
        # 读入log文件(作为普通文本附件读入)
        with open(log_cache_name, 'r') as l:
            processing_log = l.read()
            l.close()
        if processing_log[0] is not '*':
            print("processing log title erro")
        else:
            os.rename(log_cache_name, file_name + '.log')
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

    # 确定邮件题目
    mail_title = '[LOG] ' + file_name

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

    # 附件1：processing_log
    log_part = MIMEText(processing_log, 'base64', 'utf-8')
    log_part["Content-Type"] = 'application/octet-stream'
    file = file_name + log_type
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