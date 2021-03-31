Pytorch迁移学习的资料

简单说一下迁移学习，一般来说有几种做法：
1。迁移某一部分层过来
2。对某一部分层进行冻结，从而使他们保持原参数与作用，改变/新增部分层进行训练，使得网络有新的功能
3。对整个网络在新的数据上训练，迁移学习此时只是获得一个更好的初始化参数


流程资料
详细理论+使用场景的解析
http://blog.itpub.net/29829936/viewspace-2641919/

官网
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Pytorch学习(十二)—迁移学习Transfer Learning
https://www.jianshu.com/p/d04c17368922



代码细节

1.保存/载入模型的参考资料 
https://blog.csdn.net/strive_for_future/article/details/83240081

2.Pytorch模型迁移和迁移学习,导入部分模型参数
 https://blog.csdn.net/lu_linux/article/details/113373016

3.冻结与解冻层
https://www.zhihu.com/question/311095447/answer/589307812

4.pytorch中存储各层权重参数时的命名规则
https://blog.csdn.net/u014734886/article/details/106230535
