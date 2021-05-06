#use chinese
images中有以下几种，分别代表了网络的处理的复杂程度的几大阶段
（1）train_simple/val_simple（只处理了第一个域共1852张图片，将其按85：15分成train/val集，随机分配shuffle）
分别：
train_simple:1575张
val_simple:277张
（2）train_alldata_simple/val_alldata_simple(所有图片都被使用，两个域都按照85：15分成train/val集，随机分配shuffle）
分别：
train_alldata_simple:1575+770=2345张
val_alldata_simple:277+135=412张
（3）train_all_data（所有图片都充当训练集，尚未被使用）
（4）gantrain_second_domain(只使用第二个域的图片，在跨域的fintune中被使用，但是没有使用标签，相当于一个无监督的跨域）

###########################################################
注意：有三张图片中没有标注，我将其进行删除：
569
648
1552
所有模糊的图片都在训练集中