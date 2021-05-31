相关文件：
切割处理：cut8.py
关节图片数据处理：Process_image_1.py
网络定义及主程序运行：main.py

其中，将关节图片按类别存入相应文件夹时，需要更改Process_image_1中相关路径：
读取训练集更改train_path，测试集更改test_path；
分类存入文件夹的保存路径，椎间盘（disc_data）需更改save_path1，锥体（ver_data）需更改save_path2
另外注意需要提前设置好disc_data文件夹及v1-v5五个类别的文件夹，二分类同理

如要进行训练过程，在main.py的line 31更改dataset路径，即存放上一步切割好的disc_data和ver_data
选择disc_data即进行椎间盘五分类，并把line 35的num_class改为5，line59设为nn.Linear(256,5)
选择ver_data进行锥体二分类，与上同理

训练过程中存放训练好的模型，在main.py的line 143，进行五分类时存入models_5class文件夹
二分类同理，可更改相关路径

可调整train函数的参数值epoch（默认50个），运行main.py即可观测ResNet分类训练过程
训练结束后训练及测试的准确率变化过程会存入Acc_result.png


#---------注意---------#
在最终的connect_test_1中,我们首先利用训练好的UNet对图片进行定位，以及切割、写入文件夹等操作，
其中写入文件夹需调用Process_image_1.py的process_test函数，与训练过程的process不同——此时存放的是利用UNet定位切割的关节图而非txt中的给定坐标
需更改Process_image_1.py的save_path3和save_path4，同样需提前设置好valid文件夹及其下面各类（v_）文件夹

如需观测最终定位+分类结果，请将main.py的line 33的"test"改为"valid"



