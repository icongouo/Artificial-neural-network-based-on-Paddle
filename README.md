# Artificial-neural-network-based-on-Paddle

# 猫咪分类
## 项目地址：
期末考核-猫咪分类：https://aistudio.baidu.com/projectdetail/7289480?contributionType=1&sUid=5559369&shared=1&ts=1705205156774  
版本选择“草稿”或者“猫咪分类V1”。
## 项目结构：
- 引入依赖
- 数据集准备
    - 数据预处理
- ResNet50网络的搭建
- ViT的搭建
- 模型训练
    - ViT训练
    - ResNet训练
- 模型评估
    - ViT模型训练可视化及测试结果
    - ResNet训练可视化及测试结果

## 文件结构：
```
├── data                        // 数据集位置
├── visualdl_cnn                // CNN训练日志
├── visualdl_vit                // ViT训练日志
│
├── work                      // 训练权重保存路径
│   |── cnn
│   |   └── final.pdparams    // cnn权重保存
│   └── vit
│       └── final.pdparams    // vit权重保存
├── main.ipynb                //代码所在位置
├── resnet50.model            //训练前加载的预训练权重
└── ViT_base_patch16_384_pretrained.pdpaprams
```
## 测试代码
在项目结构中的“模型评估”部分：在注释处选择需要加载的模型权重。
```python
# model = VisionTransformer(
#         patch_size=16,
#         class_dim=12,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         mlp_ratio=4,
#         qkv_bias=True,
#         epsilon=1e-6)

model = ResNet(block=BottleneckBlock, num_classes=12)

pred_model = paddle.Model(model)
# pred_model.load("work/vit/final")
pred_model.load("work/cnn/final")
```
注：如果想直接进入测试环节，您需要先运行对应网络模型的搭建代码、“引入依赖”部分以及数据预处理部分，再运行“模型评估”部分代码。测试结果（可视化及验证集结果）在项目最底部以及报告中已经有所体现。（两个网络模型均在验证集上正确率超过90%，在95%左右）  
扩展部分在另一个项目，在readme后续部分会提及。  
<br />
<br />


# 新闻标题分类
## 项目地址
期末考核-新闻标题分类：https://aistudio.baidu.com/projectdetail/7394878?contributionType=1&sUid=5559369&shared=1&ts=1705208113358  
版本选择“草稿”或者“新闻标题分类”。
## 项目结构
- 数据集查看以及预处理
- 模型搭建
    - CNN模型搭建
    - Transformer模型搭建
- 模型训练
- 模型评估
    - CNN网络的训练可视化及测试结果
    - Transformer网络的训练可视化及测试结果

## 文件结构
```
├── data                        // 数据集位置
├── visualdl_cnn                // CNN训练日志
├── visualdl_transformer        // Transformer训练日志
│
├── work                      // 训练权重保存路径
│   |── cnn
│   |   └── final.pdparams    // cnn权重保存
│   └── trans
│       └── final.pdparams    // trans权重保存
├── main.ipynb                //代码所在位置
├── dev.txt                   //训练前加载的预训练权重
└── train.txt
```

## 测试代码
与猫咪分类任务类似，测试代码位于项目结构的“模型评估”处，且根据注释选择需要测试的模型。  
如果想要进行测试，您需要先运行数据预处理部分以及模型搭建部分。（建议您注释掉训练部分的代码后，一键运行所有cell，因为有些地方我分开了很多cell，逐个运行不方便）  
本项目的CNN与Transformer网络的正确率均达到90%以上，其中TextCNN达到了99%。  
扩展部分在另一个项目，在readme后续部分会提及。  
<br />
<br />

# 昆虫目标检测
## 项目地址：
期末考核-昆虫目标检测：https://aistudio.baidu.com/projectdetail/7399819?contributionType=1&sUid=5559369&shared=1&ts=1705233209060  
版本建议选择“草稿”，但项目较大（因为数据集已经解压完毕） 
## 项目结构
- 评估指标——交并比
- 数据集和数据预处理方法
    - 读取数据集标注信息
    - 数据读取和预处理
- 单阶段目标检测模型YOLO-V3
    - 模型设计思想
    - 产生候选区域
    - 卷积神经网络提取特征
    - 根据输出特征计算预测框位置
    - 损失函数
    - 多尺度检测
    - 开启端到端训练
    - 预测（模型效果及可视化展示）
## 文件结构
```
├── data                        // 数据集位置
│
├── work                      // 训练权重保存路径
│   |── insects               //解压后的数据集，具体的文件结构见项目
│   |  
│   └── yolo_epoch50.pdparams //yolov3训练了五十轮的权重
│      
├── main.ipynb                //代码所在位置
├── pred_result.json          //存放模型预测的结果
└── yolo_epoch49.paparams     //加载50轮权重后训练50轮的权重
```

## 测试代码
测试代码位于项目结构中的“预测”部分，但您需要运行除了“开启端到端”部分的大多数部分的cell，建议先到“开启端到端训练”部分，将该部分代码注释，再点击运行全部cell。

<br />
<br />

# 猫咪分类——扩展部分
## 项目地址
扩展--GAN：https://aistudio.baidu.com/projectdetail/7401332?sUid=5559369&shared=1&ts=1705210320420  
## 项目结构
- 引入依赖及数据预处理
- 定义生成器和判别器类
- 训练过程
- 评估过程
## 测试代码
测试代码位于“评估过程”部分，要进行测试，首先要运行“引入依赖及数据预处理”和“定义生成器和判别器类”部分，再运行测试代码，加载位于work目录下的模型权重，进行测试。  
（本扩展项目达到的准确率仅有57%）  
<br />
<br />

# 新闻标题分类——扩展部分
## 项目地址
该项目我没使用paddle，使用了pytorch框架进行实现，项目已打包发送给助教学长。

## 文件结构
```
├── data                        // 存储词表等数据
├── dataset                     // 训练集、验证集
├── log                         // 训练日志
│
├── saved_dict                  // 训练权重保存路径
│   
├── main.py                     //训练代码
├── processing_data.py          //数据预处理代码
└── test.py                     //测试代码所在位置
```
## 测试代码
测试代码位于上述文件结构中的“test.py”，直接运行即可在终端输出loss以及accuracy。
