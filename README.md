# 项目简介

该项目使用Lora等微调方法,对qwen2.5-vl-7b-instruct模型进行微调

# 项目环境

详细可见requirements.txt文件

# 训练数据集

1. coco_2014_caption(共计25万条数据)。使用脚本：download_coco_2014.py进行下载处理

# 训练情况

| 模型        | 训练方法             | 模型总参数量 | 可训练参数量 | batch_size | 显存占用 | Lora Rank | Lora Alpha |
|-------------|------------------|--------|--------|------------|------|-----------|------------|
| qwen2.5-vl-7b-instruct | Lora             | None   | None   | None       | None  | None        | None     |


# 注意事项

1. 该项目是使用lora方法进行的微调，暂时没有量化，后续可以根据资源进行选配。

# 未来工作


# 效果展示
