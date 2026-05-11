# 冷启动数据集建议

推荐用于审核系统冷启动的公开数据集：

## 文本
- Jigsaw Toxic Comment Classification Challenge（Kaggle）
- Hate Speech and Offensive Language Dataset（Davidson 等）

## 图片
- Open Images（暴力/武器相关类别子集）
- Kaggle 上的 NSFW 抓取数据集（仅建议 PoC 使用，生产前务必核对许可证）

## 视频
- RWF-2000（暴力检测）
- XD-Violence（异常/暴力视频事件）

PoC 导入策略：

1. 将每条样本转换为统一元数据与摘要文本。
2. 使用向量模型生成 embedding。
3. 入库到 Milvus，已人工确认的数据设置 `human_verified=true`。

在线自举策略：

1. 自动审核结果满足 `confidence>=0.9` 且复核通过的样本，设置 `human_verified=false`、`source=weak_label` 后入库。
2. 人工审核回流结果统一标记为 `human_verified=true` 并入库。
3. 按天执行离线任务，使用已验证样本重训阈值策略与预打标模型。
