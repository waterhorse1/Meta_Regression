由于做了较多实验，所以这里把每一个py文件和ipynb文件的作用和功能阐述：
data文件夹：存放生成regression任务的py文件
model文件夹：存放训练好的model
multi_result_files文件夹：存放训练测试结果的数据文件夹

（部分ipynb可能需要jupyter notebook显示）
RNN_encoder_test.ipynb: HSML一文中RNN encoder的实现，用来确定通过任务重构loss的embedding能不能把不同种类loss区分开来，目前来看，即使是两类函数，类似AE的模型仍然无法将二者区分开来。
embedding.ipynb HSML一文中全连接encoder的实现，结果同上。后面有一些杂七杂八的代码可以不管。
vae_clustering.ipynb 一部聚类的VAE方法，同样无法将不同函数通过VAE的loss区分开来
HSML.ipynb 在写的HSML代码，可以先不管

cavia部分以及选位置部分代码
cavia.py:cavia+选位置
cavia_do.py cavia+随机选位置，证明encoder来task-specific选位置确实有效
cavia_model_back.py：包含encoder模型，place模型以及cavia的模型定义的py文件
caiva_rnn.py task_embedding_gru（见聚类py代码） + 选位置（或者直接定context 参数值）py文件
cavia_ori.py与caiva_backup.py cavia代码的最原始代码，baseline
cavia_dropout.py 和cavia_dropout2.py都是最开始对于不同位置直接选择不同层的参数的代码，效果也不好（这两个代码可能没法直接跑了因为我改了一些东西）
cavia_em_dp.py cavia+选位置梯度下降+选位置直接赋值context_parameters,效果与单纯选位置相同。
cavia_model.py：可以暂时不管这个代码

cavia_recon.py 训练过程中引入重构loss再加选位置的py文件
gumbel_sample.py：包含通过logits进行gumbel采样，选位置的代码

maml代码：
maml_model.py 定义了maml的模型
maml.py 跑maml代码，baseline
maml_dropout.py maml+比cavia更多的部分网络中部分层的部分参数（如全连接第一层的bias或者weight），来做cavia的方式，发现效果仍存在，所以cavia的方式其实可以有很多种。

ELLA部分代码
ella.py 跑ella类代码
ella_copy.py 用于备份

一些聚类py代码
task_embedding.py 内包含一步采用VAE聚类的embedding代码，曾与选位置代码一起测试，无聚类效果
task_embedding_gru.py 内包含采用RNN encoder聚类的embedding代码，曾与选位置代码一起测试，无聚类效果

model.py 简单粗暴的写了个参数放入模型的model

一下为原cavia文章的代码，除了main文件，argument文件以外其他可以不用管
tasks_celeba.py：celeba数据集
tasks_sine.py：sine函数数据集
以上为不用管，定义数据生成的py文件在data文件夹里

utils.py：一些函数例如设置随机种子，保存和读取一些文件
logger.py:定义类来记录模型，训练过程，验证过程以及测试过程的数值。
main.py 用来跑代码
argument.py 设置代码中各种参数

跑代码：可以通过调节arguement中num_context_params来调节每一层输入的context参数个数，注意跑maml的时候把这个参数设成数值例如5才能正常跑
跑maml.py： python3 main.py --maml --num_context_params 0
cavia源代码： python3 main.py --cavia_ori
cavia+选位置 python3 main.py

注意代码里面由于我现在在测试embedding能不能区分开来，data文件夹中task_multi我只选了两种函数且这两种函数的变化非常小，所以如果要进行严谨实验请参照HSML里面的函数设定，具体可见HSMLpaper。
（具体来说就是修改一下
data.task_test_quadratic import quadratic 中quadratic的设定和
data.task_test_sine import sine 中sine的变量设定 比较简单就不再赘述