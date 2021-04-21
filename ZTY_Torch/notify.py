"""
版本 3月 13日 01：30  这个文件用来自动发送log等
这个文件用来自动发送 输出log + 性能监控log + 追加的文件 到指定邮箱列表中
作者：吕尚青 张天翊 吴雨卓
-------------------------------------------------------------------------------
需要监控的程序可在启动时用如下代码调用本功能：

import sys
import os
# 将当前目录和父目录加入路径，使得文件可以调用本目录和父目录下的所有包和文件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import notify

# 程序代码
...
# 在邮件里新增一段文本，可多次用
notify.add_text("whatever u want to say")

# 追加邮件附件，可以是文件/文件夹的文件路径（会自动zip），只需要在任意位置调用这个函数即可，可多次用,添加多个附件
notify.add_file(file_name）
...
notify.send_log()
# 选择需要发送邮件的邮箱，空则为default list
# 在自己代码中的任意位置调用就行。注意：如果不调用，则邮件中的程序名为default，且自动发送给默认邮箱

-------------------------------------------------------------------------------
公邮：foe3305@163.com
密码：ddd888
如果想只发给自己，就把自己邮箱写进去：在自己代码中任意位置使用，以最后一次调用为准
notify.send_log(“1111@111.com”)
如果要发给多人，请传入一个包含多个str的元组/列表。
不写发给谁的话，默认会发给一个默认列表中的所有人，需要加入默认列表私聊zty。公邮在默认列表中。

-------------------------------------------------------------------------------
说明：
输出监控日志格式：
*****************LOG_Cache_2020_12_31_01_01*****************
内容
内容
内容

start time: 2020_12_31  01:01:14
end time: 2020_12_31  01:02:04
source: 服务器名
-------------------------------------------------------------------------------
性能监控日志格式：
============================================
监控开始时间:    2021-01-29 03:24:34
采样间隔(s):  5  | 计算均值写入日志间隔(s):   300
============================================
时间: 2021-01-29 03:24:39   | CPU平均占用率: 3.86  | 内存占用率: 75.4
============================================
监控结束时间:    2021-01-29 03:24:39
平均CPU占用率:   3.86  | 平均内存占用率:  75.4
最大CPU占用率:   3.86  | 最大内存占用率:  75.4

-------------------------------------------------------------------------------
2021.1.1 17:00  更新内容：本地日志文件保存在程序目录下的log文件夹内
2021.1.2 11:00  修复了计算均值写入日志间隔的bug
2021.1.29 3:30  修复了程序出错时无法正常发邮件的问题
                增加了自动清空旧日志的功能（可设置）
2021.2.2 1.30   修复了程序名显示错误的问题
-------------------------------------------------------------------------------
注意这个函数的顺序很重要，不要改顺序!!!!!!
维护工作：
 - 生成log部分/追加附件，吕尚青，张天翊
 - 监控log部分，吕尚青
 - 发送log部分，吴雨卓
-------------------------------------------------------------------------------
"""

import threading
import os
import psutil
import re
import socket
import sys
import time
import shutil
import zipfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class NotifyBackend(threading.Thread):
    """
    后台运行的notify线程
    功能：
     - 监控服务器性能参数
     - 监控主进程是否存活
     - 主进程结束后打包日志数据并发送给指定邮箱，之后自我了断
    输入：
     - log_folder_path     日志文件夹目录
     - mail_host
     - mail_user
     - mail_pass
     - mail_list           邮件接收人列表
    依赖：
     - class Logger
    """

    def __init__(self, log_root_path, log_folder_name, mail_host, mail_user, mail_pass, mail_list):
        threading.Thread.__init__(self, name='notify')

        # 用户定义监控参数
        self.report_time = 300  # 每次计算均值写入日志的间隔时间 300s
        self.sample_time = 5  # 每次监控采样的间隔时间 5s
        self.max_log_under_root_path = 5  # 同一个日志来源的最大日志数

        # 定义全局变量
        self.mail_host = mail_host
        self.mail_user = mail_user
        self.mail_pass = mail_pass
        self.mail_list = mail_list  # 邮件发送对象列表

        call_func_name = 'default'
        self.log_folder_name = log_folder_name  # 日志文件夹名（时间）
        self.log_root_path = log_root_path  # 日志根目录
        self.log_folder_path = os.path.join(log_root_path, call_func_name, log_folder_name)  # 日志文件夹目录
        self.finish_process = 0  # 进程正常结束后赋值1
        self.monitor_process = False  # 监控线程
        self.additional_explain = ''  # 额外说明，一般包含压缩文件无法找到等信息。如果被赋值，会被追加到Trans_Body.log对应的邮件正文中
        self.start_time = time.time()  # 记录监控开始运行时间

        # 启动服务器性能监控
        self.start_monitor(self.log_folder_path, log_name='Server_Status.log',
                           report_time=self.report_time, sample_time=self.sample_time)

    def run(self):
        """
        检查主线程是否存活
        :return:
        """
        while 1:
            # 检查主进程是否结束，每self.sample_time秒查一次
            for i in threading.enumerate():
                if i.name == "MainThread" and not i.is_alive():  # 主进程结束后开始料理后事
                    self.stop_monitor()  # 结束服务器性能监控
                    self.send_email()
                    return
            time.sleep(self.sample_time)

    '''
    ****************************************
    服务器性能监控函数
    ****************************************
    '''

    def start_monitor(self, log_dir, log_name='server_status.log', report_time=300, sample_time=5):
        """
        启动函数
        :param log_dir:
        :param log_name:
        :param report_time:
        :param sample_time:
        :return:
        """
        self.monitor_process = threading.Thread(target=self.server_monitor_process, daemon=True,
                                                args=(log_dir, log_name, report_time, sample_time))
        self.monitor_process.start()

    def stop_monitor(self):
        if bool(self.monitor_process):
            self.finish_process += 1
            self.monitor_process.join()  # 等待monitor_process线程完成

    def server_monitor_process(self, log_dir, log_name='server_status.log', report_time=300, sample_time=5):
        """
        主函数
        :param log_dir: 日志保存目录
        :param log_name: 日志文件名
        :param report_time: 每次计算均值写入日志的间隔时间
        :param sample_time: 每次监控采样的间隔时间
        :return:
        """
        next_time_to_report = report_time

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.write_information_to_log(log_dir, log_name, info_type='init', report_time=report_time,
                                      sample_time=sample_time)
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
            if self.finish_process == 1:
                cpu_avg_list.append(self.calc_avg_cpu_usage_percentage(cpu_list))
                mem_avg_list.append(self.calc_avg_mem_usage_percentage(mem_list))
                cpu_max_list.append(self.calc_max_cpu_usage(cpu_list))
                mem_max_list.append(max(mem_list))
                self.save_server_log(cpu_avg_list[-1], mem_avg_list[-1], log_dir, log_name)

                cpu_avg = self.calc_avg_mem_usage_percentage(cpu_avg_list)
                mem_avg = self.calc_avg_mem_usage_percentage(mem_avg_list)
                cpu_max = self.calc_avg_mem_usage_percentage(cpu_max_list)
                mem_max = self.calc_avg_mem_usage_percentage(mem_max_list)
                self.write_information_to_log(log_dir, log_name, info_type='finish',
                                              cpu_avg=cpu_avg, mem_avg=mem_avg, cpu_max=cpu_max, mem_max=mem_max)
                return 0

            # 正常保存
            elif next_time_to_report <= 0:
                cpu_avg_list.append(self.calc_avg_cpu_usage_percentage(cpu_list))
                mem_avg_list.append(self.calc_avg_mem_usage_percentage(mem_list))
                cpu_max_list.append(self.calc_max_cpu_usage(cpu_list))
                mem_max_list.append(max(mem_list))
                self.save_server_log(cpu_avg_list[-1], mem_avg_list[-1], log_dir, log_name)
                cpu_list = []
                mem_list = []
                next_time_to_report = report_time

    def calc_avg_cpu_usage_percentage(self, cpu_usage_list_divided_by_time):
        avg_cpu_usage = 0
        cnt = 0
        for _sample in cpu_usage_list_divided_by_time:
            for single_cpu_percentage in _sample:
                avg_cpu_usage += single_cpu_percentage
                cnt += 1
        avg_cpu_usage = avg_cpu_usage / cnt
        return round(avg_cpu_usage, 2)

    def calc_max_cpu_usage(self, cpu_usage_list_divided_by_time):
        max_cpu_usage = 0
        for _sample in cpu_usage_list_divided_by_time:
            avg_usage = 0
            for single_cpu_percentage in _sample:
                avg_usage += single_cpu_percentage
            avg_usage /= len(_sample)
            if avg_usage > max_cpu_usage:
                max_cpu_usage = avg_usage
        return round(max_cpu_usage, 2)

    def calc_avg_mem_usage_percentage(self, mem_usage_list_divided_by_time):
        avg_mem_usage = 0
        for _sample in mem_usage_list_divided_by_time:
            avg_mem_usage += _sample
        avg_mem_usage = avg_mem_usage / len(mem_usage_list_divided_by_time)
        return round(avg_mem_usage, 2)

    def save_server_log(self, cpu_usage, mem_usage, log_dir, log_name):
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        format_save = r'时间: %s   | CPU平均占用率: %s  | 内存占用率: %s  ' % \
                      (now_time, str(cpu_usage), str(mem_usage))
        with open(os.path.join(log_dir, log_name), mode="a", encoding="utf-8") as f:
            f.write(format_save + '\n')
            f.close()

    def write_information_to_log(self, log_dir, log_name, info_type, report_time=60, sample_time=5,
                                 cpu_avg='', mem_avg='', cpu_max='', mem_max=''):
        """
        在日志的开始或者结尾写入统计信息
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
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        if info_type == 'init':
            status_statement = '============================================\n' \
                               '监控开始时间:    %s\n' \
                               '采样间隔(s):  %s  | 计算均值写入日志间隔(s):   %s  \n' \
                               '============================================' % \
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

    '''
    ****************************************
    文件读取及发送函数
    ****************************************
    '''

    def send_email(self):
        """
        发送邮件
        :return:
        """

        # 日志格式
        log_type = ".rtf"

        log_cache_path = os.path.join(self.log_folder_path, 'Log_Cache.log')
        trans_file_path = os.path.join(self.log_folder_path, 'Trans_File.log')
        trans_body_path = os.path.join(self.log_folder_path, 'Trans_Body.log')
        settings_path = os.path.join(self.log_folder_path, 'Settings.log')
        trans_file_zip_path = os.path.join(self.log_folder_path, 'Temp_Zip_File')  # 待传文件压缩包位置
        server_status_path = os.path.join(self.log_folder_path, 'Server_Status.log')  # 服务器性能监控日志地址
        func_name_path = os.path.join(self.log_folder_path, 'Func_Name.log')

        # 配置邮箱信息
        if os.path.exists(settings_path):
            self.mail_list = []
            for mail_recv in open(settings_path, 'r'):
                self.mail_list.append(re.sub(r'\n', '', mail_recv))

        # 确定时间
        time_start = self.start_time
        time_end = time.time()

        # 确定服务器来源：
        source_server = self.get_host_name()

        # 宣布完成并总结运行情况
        print('\n' + '=' * 60)
        print("Processing finished !")
        print("start time:", time.strftime('%Y_%m_%d  %H:%M:%S', time.localtime(time_start)))
        print("end time:", time.strftime('%Y_%m_%d  %H:%M:%S', time.localtime(time_end)))
        print("source:", source_server)

        # 定log文件名
        try:
            with open(func_name_path, 'r') as l:
                call_func_name = l.read()
                l.close()
        except Exception as e:
            print('func send_log() has not called, use default func name: ', e)
            call_func_name = 'default'
        processing_log_name = call_func_name + '__' + time.strftime('%Y_%m_%d-%H_%M_%S',
                                                                    time.localtime(time_start)) + '_log'

        print("\nPreparing the email with auto log file :\n", processing_log_name, '\nas ',
              log_type)

        # 确定邮件题目
        mail_title = '[' + source_server + '  LOG] ' + processing_log_name

        # 组织email内容
        message = MIMEMultipart()
        message['Subject'] = mail_title
        message['From'] = self.mail_user

        # 如果是收件人列表，做编码处理
        if len(self.mail_list) > 1:
            message['To'] = ";".join(self.mail_list)
        elif len(self.mail_list) == 1:  # 如果只是一个邮箱， 就发到这个邮箱
            message['To'] = self.mail_list[0]
        else:
            print("mail_list problem occur!")
            return -1

        # 处理邮件正文文本
        running_info = "start time: %s \nend time: %s \nsource: %s \n=================\n\n" % (
            time.strftime('%Y_%m_%d  %H:%M:%S', time.localtime(time_start)),
            time.strftime('%Y_%m_%d  %H:%M:%S', time.localtime(time_end)),
            self.get_host_name()
        )

        if os.path.exists(trans_body_path):
            with open(trans_body_path, 'r', encoding='UTF-8') as l:
                trans_body_content = l.read()
                l.close()
            running_info += ('\n' + trans_body_content)
        message.attach(MIMEText(running_info, 'plain', 'utf-8'))

        # 处理追加附件
        self.prepare_trans_file()  # 压缩用户指定传输的文件（如果有的话）
        if os.path.exists(trans_file_zip_path):
            for zip_file in os.listdir(trans_file_zip_path):
                zip_file_full_path = os.path.join(trans_file_zip_path, zip_file)
                with open(zip_file_full_path, 'rb') as Af:
                    file = Af.read()
                try:
                    Af.close()
                    # 添加附件
                    log_part = MIMEText(file, 'base64', 'utf-8')
                    log_part["Content-Type"] = 'application/octet-stream'
                    # 这里的filename可以任意写，写什么名字，邮件中显示什么名字
                    log_part["Content-Disposition"] = 'attachment; filename="%s"' % zip_file
                    message.attach(log_part)
                except:
                    print("Erro occur in adding additional file:", zip_file)
                else:
                    print("An additional file has been added to the mail:", zip_file)

        # 阻断log生成
        # （stdout在import notify的时候就已经被重定义为Logger类，所以这里直接调用Logger的函数）
        sys.stdout.close_log_and_put_back()
        # sys.stderr.close_log_and_put_back()   # stderr不能加这行代码，原理我暂时不清楚    ——LSQ

        # 调取程序的print输出: processing_log
        try:
            # 读入log文件(作为普通文本附件读入)
            with open(log_cache_path, 'r', encoding='UTF-8') as l:
                processing_log = l.read()
                l.close()
            if processing_log[0] is not '*':
                print("processing log title erro")
        except Exception as e:
            print("processing log status erro: ", e)
            return -1
        else:
            print("processing log catched")

        # 调取服务器日志: server_log
        try:
            with open(server_status_path, 'r', encoding='UTF-8') as f:
                server_log = f.read()
                f.close()
            if server_log[0] is not '=':
                print("server log title erro")
        except Exception as e:
            print("server log status erro: ", e)
            return -1
        else:
            print("server log catched")

        try:
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
            file = 'server_status' + log_type
            log_part[
                "Content-Disposition"] = 'attachment; filename="%s"' % file  # 这里的filename可以任意写，写什么名字，邮件中显示什么名字
            message.attach(log_part)

            # 实例化，也是登录的过程
            smtp = smtplib.SMTP_SSL(self.mail_host, timeout=3000)
            smtp.ehlo(self.mail_host)
            smtp.login(self.mail_user, self.mail_pass)
            smtp.sendmail(self.mail_user, self.mail_list, message.as_string())
            smtp.quit()
            print('发送log邮件成功，title: ', mail_title)
            print('如果没有，看看垃圾箱:)')

            # 移动日志并删除过时数据
            try:
                sys.stderr.close_log_and_put_back()  # 关闭告警记录
            except:
                pass

            new_root_path = os.path.join(self.log_root_path, call_func_name)
            new_folder_path = os.path.join(new_root_path, self.log_folder_name)
            if not os.path.exists(new_root_path):
                os.mkdir(new_root_path)
            self.delete_obsolete_log(new_root_path)
            shutil.move(self.log_folder_path, new_root_path)
        except Exception as e:
            print('邮件发送失败: ', e)

    def prepare_trans_file(self):
        """
        准备用于邮件传输的文件（如果有的话）
        日志保存结构：
         - Log_Cache.log        保存所有print日志
         - Server_Status.log    保存监控信息（在NotifyBackend类中定义）
         - Trans_Body.log       保存文字说明，发送邮件时作为邮件正文
         - Trans_File.log       保存待传文件地址
         - Temp_Zip_File        文件夹，保存压缩包（在NotifyBackend类中定义）
        :return:
        """
        # 读取需要传输的文件地址
        trans_file_log_path = os.path.join(self.log_folder_path, 'Trans_File.log')  # Trans_File.log文件地址

        if os.path.exists(trans_file_log_path):  # 如果Trans_File.log文件存在，则逐行读取

            # 创建压缩目录Temp_Zip_File
            trans_file_zip_path = os.path.join(self.log_folder_path, 'Temp_Zip_File')
            if not os.path.exists(trans_file_zip_path):
                os.mkdir(trans_file_zip_path)

            # 读取所有文件并压缩存入Temp_Zip_File
            for file_path in open(trans_file_log_path, 'r'):
                file_path = re.sub(r'\n', '', file_path)
                full_path = os.path.join(os.getcwd(), file_path)
                if os.path.exists(full_path):
                    zip_file_name = re.findall(r'[^/\\]+$', file_path)[0]  # zip文件名（和原文件名一致）
                    zip_err = self.zipDir(full_path, os.path.join(trans_file_zip_path, zip_file_name))  # 压缩待传文件并保存
                    if zip_err:
                        print('zip error! details below: \n', zip_err)
                else:
                    print('cannot zip file: ', file_path)

    def zipDir(self, dirpath, outFullPath):
        """
        压缩指定文件夹到指定路径
        :param dirpath: 目标文件夹路径:1212/12/c
        :param outFullPath:  'aaa/bbb/c.zip'
        :return: 无
        """
        try:
            zip = zipfile.ZipFile(outFullPath + '.zip', 'w', zipfile.ZIP_DEFLATED)
            # 目录：递归压缩
            if os.path.isdir(dirpath):
                for path, dirnames, filenames in os.walk(dirpath):
                    # 去掉目标和路径，只对目标文件夹下边的文件及文件夹进行压缩（包括父文件夹本身）
                    parent_path = os.path.abspath('.')  # 父目录
                    fpath = path.replace(dirpath, '')  # 子目录（文件目录）
                    for filename in filenames:
                        zip.write(os.path.join(path, filename),
                                  os.path.join(fpath, filename))
                zip.close()
            # 文件：直接压缩
            elif os.path.isfile(dirpath):
                zip.write(dirpath, re.findall(r'[^/\\]+$', outFullPath)[0])
                zip.close()
        except Exception as e:
            return e
        return 0

    def delete_obsolete_log(self, log_root_path):
        """
        检查并删除过早的日志文件
        :return:
        """
        create_time_dict = {}
        create_time_list = []
        for file_name in os.listdir(log_root_path):
            c_time = int(time.mktime(time.strptime(file_name, '%Y_%m_%d-%H_%M_%S')))
            create_time_list.append(c_time)
            create_time_dict[c_time] = file_name
        if len(create_time_list) >= self.max_log_under_root_path - 1:
            create_time_list.sort()
            for c_time in create_time_list[:-self.max_log_under_root_path + 1]:
                shutil.rmtree(os.path.join(log_root_path, create_time_dict[c_time]))
                print('obsolete log deleted: ', create_time_dict[c_time])

    def get_host_name(self):
        """
        获取本机hostname
        :return:
        """
        return socket.gethostname()


class NotifyFrontend:
    """
    主程序import notify时调用的notify类，被调用时会启动NotifyBackend作为后台线程
    不单独作为线程，和主程序同步结束，只进行文件操作
    功能：
     - 将主程序及其所有线程的print输出导入到日志
     - 根据命令存储指定文本和文件
    日志保存结构：
     - Log_Cache.log        保存所有print日志
     - Server_Status.log    保存监控信息（在NotifyBackend类中定义）
     - Trans_Body.log       保存文字说明，发送邮件时作为邮件正文
     - Trans_File.log       保存待传文件地址
     - Temp_Zip_File        文件夹，保存压缩包（在NotifyBackend类中定义）
    依赖：
     - class Logger
     - class NotifyFrontend
    """

    def __init__(self, log_root_path, mail_host, mail_user, mail_pass, mail_list, max_log_cnt=5):
        ## 设置参数
        # call_func_name = sys._getframe(1).f_code.co_filename.split('/')[-1]  # 获取程序名作为主目录
        # call_func_name = re.findall(r'[^/\\]+$', call_func_name)[0].split('.py')[0]

        ## 文件保存参数
        call_func_name = 'default'
        self.log_root_path = os.path.join(log_root_path, call_func_name)  # 日志主目录 = 日志根目录 + 程序名
        self.max_log_under_root_path = max_log_cnt  # 日志根目录下最多日志数，如果超过会自动删除过早的日志文件夹

        log_folder_name = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime(time.time()))  # 当前时间作为子文件夹名
        self.log_folder_path = os.path.join(self.log_root_path, log_folder_name)

        # 所有运行中代码的pring输出都保存在此文件
        self.log_cache_path = os.path.join(self.log_folder_path, 'Log_Cache.log')
        # 待传输文件地址保存文件
        # 传输原理: 调用notify.add_file(file_name）时，将file_name的地址写入该位置，程序结束后NotifyBackend读取对应文件并附加到邮件内
        self.trans_file_path = os.path.join(self.log_folder_path, 'Trans_File.log')
        # 邮件主体文字保存文件。调用notify.add_text("...")时，文字保存在此位置
        self.trans_body_path = os.path.join(self.log_folder_path, 'Trans_Body.log')
        # 设置保存位置。此处保存邮件的发送对象
        self.settings_path = os.path.join(self.log_folder_path, 'Settings.log')
        self.func_name_path = os.path.join(self.log_folder_path, 'Func_Name.log')

        # 创建目录
        if not os.path.exists(log_root_path):
            os.mkdir(log_root_path)
        if not os.path.exists(self.log_root_path):
            os.mkdir(self.log_root_path)
        if os.path.exists(self.log_folder_path):
            os.rmdir(self.log_folder_path)
        os.mkdir(self.log_folder_path)

        self.delete_obsolete_log()  # 删除过时日志文件

        # 重定向pring输出至文件，程序结束后自动退出
        sys.stdout = Logger(self.log_cache_path, path=os.getcwd())  # 正常输出
        sys.stderr = Logger(self.log_cache_path, path=os.getcwd())  # 告警输出
        # sys.stdout = open(self.log_cache_path, 'w')

        # 写入日志头
        fileName = time.strftime('LOG_Cache_' + '%Y_%m_%d_%H_%M', time.localtime(time.time()))
        print(fileName.center(60, '*'))

        # 启动notify进程
        notify_backend_thread = NotifyBackend(log_root_path, log_folder_name, mail_host=mail_host, mail_user=mail_user,
                                              mail_pass=mail_pass, mail_list=mail_list)
        notify_backend_thread.start()
        print('notify started')

    def delete_obsolete_log(self):
        """
        检查并删除过早的日志文件
        :return:
        """
        create_time_dict = {}
        create_time_list = []
        for file_name in os.listdir(self.log_root_path):
            c_time = int(time.mktime(time.strptime(file_name, '%Y_%m_%d-%H_%M_%S')))
            create_time_list.append(c_time)
            create_time_dict[c_time] = file_name
        if len(create_time_list) >= self.max_log_under_root_path - 1:
            create_time_list.sort()
            for c_time in create_time_list[:-self.max_log_under_root_path + 1]:
                shutil.rmtree(os.path.join(self.log_root_path, create_time_dict[c_time]))
                print('obsolete log deleted: ', create_time_dict[c_time])

    def add_a_text(self, text_input):
        with open(self.trans_body_path, 'a') as file_object:
            file_object.write(text_input + '\n')

    def add_a_file(self, file_dir):
        with open(self.trans_file_path, 'a') as file_object:
            file_object.write(file_dir + '\n')

    def send_log(self, mail_list, call_func_name):
        """
        把邮件发送地址和程序名称写入文件，供NotifyBackend调取
        :param mail_list: 邮件发送地址列表
        :param call_func_name: 主程序名
        :return:
        """
        if type(mail_list) in [list, tuple]:
            if type(mail_list) == tuple:
                mail_list = list(mail_list)
            with open(self.settings_path, 'w') as file_object:
                for mail_recv in mail_list:
                    file_object.write(mail_recv + '\n')
        elif type(mail_list) == str:
            with open(self.settings_path, 'w') as file_object:
                file_object.write(mail_list)

        with open(self.func_name_path, 'w') as file_object:
            file_object.write(call_func_name)


class Logger(object):
    """
    功能：重定义print输出至指定文件
    """

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


# 设置参数
log_root_path = 'log'  # 日志根目录
max_log_cnt = 5  # 同一个程序保留的日志个数（包括当前和历史日志），过多可能会占用主存

# 配置邮箱信息
mail_host = 'smtp.exmail.qq.com'
mail_user = 'notice@visionwyz.com'
mail_pass = '3cvPbaNucRHvNiJb'  # 腾讯企业邮箱的授权码
default_receivers = ('foe3305@163.com')

# 启动notify后台进程
notify_frontend = NotifyFrontend(log_root_path=log_root_path, mail_host=mail_host,
                                 mail_user=mail_user, mail_pass=mail_pass,
                                 mail_list=default_receivers, max_log_cnt=max_log_cnt)


def add_text(text_input, notify_frontend=notify_frontend):
    """
    设置邮件正文内容
    :param text_input: 追加的文件内容
    :param notify_frontend: notify类（不用管）
    :return:
    """
    if bool(notify_frontend):
        notify_frontend.add_a_text(text_input=text_input)


def add_file(file_dir, notify_frontend=notify_frontend):
    """
    追加邮件附件，可以是文件/文件夹（会自动zip），只需要调用这个函数即可
    :param file_dir: 追加的附件路径，可以是文件/文件夹（会自动zip）
    :param notify_frontend: notify类（不用管）
    :return:
    """
    notify_frontend.add_a_file(file_dir=file_dir)
    print(file_dir, " has been added to the mail attachment list as an additional file")


def send_log(mail_list=default_receivers, notify_frontend=notify_frontend):
    """
    设置接收邮箱，可以在代码中的任何位置设置，程序执行完成后邮件会发往最后指定的接收方
    :param mail_list: 日志发送对象，可以是str或者list类型
    :param notify_frontend: notify类（不用管）
    :return:
    """
    if bool(mail_list):
        call_func_name = sys._getframe(1).f_code.co_filename.split('/')[-1]  # 获取程序名作为主目录
        call_func_name = re.findall(r'[^/\\]+$', call_func_name)[0].split('.py')[0]
        notify_frontend.send_log(mail_list, call_func_name)