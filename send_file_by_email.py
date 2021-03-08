"""
版本 2月 23日 00：50  用来发送email的程序
维护：张天翊
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from general import notify
import time

time.sleep(1)
print("print file path please:")
file_path = input()

if os.path.exists(file_path):
    notify.add_file(file_path)

    print("write the target email path please: use blankspace for the dafult path ")
    email_path = input()

    if len(email_path) < 5:
        print("send mail to defult list")
        notify.send_log()
    else:
        print("send mail to:", email_path)
        notify.send_log(email_path)

else:
    print("file path is not exist!!!!")


