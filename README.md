# 房价预测 - XGBoost实现
基于XGBoost的房价预测模型，通过数据清洗、特征工程、K折交叉验证，将预测误差（MAE）控制在1万以内。

## 项目介绍
- 数据源：Kaggle房价预测数据集（train.csv/test.csv）
- 核心模型：XGBoost回归
- 关键优化：缺失值智能填充、高价值特征工程、异常值精准处理
- 最终目的：5折交叉验证平均MAE＜10000
- 目前效果：5折交叉验证平均MAE约等于14000

## 运行步骤
1. 安装依赖：
   ```bash
   pip install -r requirements.txt