基于resnet18{
文件中为resnet18-f37072fd.pth
因为基于别的权重（包括resnet更大的权重、别的系列前期尝试太多）训练得到的一般都超时；
且resnet本身较适合小样本学习
}
训练数据集，得到rsenet18-better 系列（未展示）
其中_1为初版本 _2增加数据处理方式   _3增加训练的epoch
之后微调，得到第13个epoch的acc最高  
好吧，过拟合


最终版本train118.py（11月8号训练出来的）采用余弦退火，细化学习率，训练6个epoch，得到resnet18-finetuned

国赛：
修改，在4轮过后学习率改为1e-5，训练8轮

最终准确率99.992%，针对特定训练集
有点乱，不整理了，讲就看吧。
