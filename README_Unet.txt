相关文件:
从txt中读取label数据:tool.py
载入数据dataset文件:firststage_Dataset.py
网络定义:U_Net.py
测试文件:U_Net_test.py
训练文件:U_Net_train.py

如需重新训练与测试模型:
只需改动main函数中的parser.add_argument()相关参数
train:先修改U_Net_train.py的main函数中的parser.add_argument()内--traindata_path与--testdata_path训练与
测试数据集的路径以及修改训练结果保存路径(代码中已有相关注释)。然后 python U_Net_train.py 命令运行程序即可
test:先修改U_Net_test.py的main函数中的parser.add_argument()内--testdata_path测试数据集的路径以及--pretrain_dir加载模型参数的路径。
然后python U_Net_test.py 命令运行程序即可

关键代码:
tool.py中的读取label数据
firststage_Dataset.py中firststage_Dataset_plus的数据增强
U_Net_train.py中的getheatmap创建高斯分布热力图


#---------以下为个人的小记录------------#
训练/测试结果:
exp.txt为多次训练的结果
.\train_Unet_sigma3_withdataaugment_move\trainsigma3dataplus文件夹中为最佳模型的训练过程的输出，其中:
Acc_result.png为训练和测试过程中，Acc(定义在半径3以内为命中)的趋势。
Acc16_result.png为训练和测试过程中，Acc16(定义在半径4以内为命中)的趋势。
EpochN_pred.png为在第N个Epoch训练中的预测图，EpochN_pred_M.png为在第N个Epoch训练中第M+1个点(从上到下)的预测图，
数据缺失问题:
168 82 164 77



