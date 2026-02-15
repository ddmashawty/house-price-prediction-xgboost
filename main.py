import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# ===================== 1. 数据预处理（彻底修复列匹配问题） =====================
# 读取数据 相对路径（确保数据文件在正确位置）
train_path = "./data/train.csv"
test_path = "./data/test.csv"
df = pd.read_csv(train_path, encoding="utf-8")
test_df = pd.read_csv(test_path, encoding="utf-8")

# ===== 核心修复1：提取原始列时直接排除标签列 =====
# 定义标签列名
target_col = "SalePrice"
# 原始数值列：排除标签列，仅保留训练集原始int/float列
original_numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target_col).tolist()
# 原始类别列：仅保留训练集原始object列
original_categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# 特征工程函数（仅构造训练/测试集都有的特征，无专属列）
def add_features(df):
    df = df.copy()
    # 1. 总生活面积 = 地上面积 + 地下室面积
    if 'GrLivArea' in df.columns and 'TotalBsmtSF' in df.columns:
        df['TotalLivingArea'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)
        if 'TotalLivingArea' not in original_numeric_cols:
            original_numeric_cols.append('TotalLivingArea')
    # 2. 房屋年龄 = 销售年份 - 建造年份
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        if 'HouseAge' not in original_numeric_cols:
            original_numeric_cols.append('HouseAge')
    # 3. 总浴室数 = 全浴室 + 半浴室*0.5
    if 'FullBath' in df.columns and 'HalfBath' in df.columns:
        df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']
        if 'TotalBath' not in original_numeric_cols:
            original_numeric_cols.append('TotalBath')
    # 4. 房间密度 = 总房间数/总生活面积
    if 'TotRmsAbvGrd' in df.columns and 'TotalLivingArea' in df.columns:
        df['RoomPerArea'] = df['TotRmsAbvGrd'] / df['TotalLivingArea'].replace(0, 1)
        if 'RoomPerArea' not in original_numeric_cols:
            original_numeric_cols.append('RoomPerArea')
    # 移除所有可能的标签列（防止混入）
    if target_col in df.columns:
        df = df.drop(target_col, axis=1)
    return df

# ===== 处理训练集 =====
# 分离标签和特征（先提标签，再做特征工程）
y = df[target_col].copy()
# 处理房价异常值（截断1%~99%分位数）
q1, q99 = y.quantile(0.01), y.quantile(0.99)
y = np.clip(y, q1, q99)
# 训练集特征工程（已移除标签列）
X = add_features(df)

# ===== 处理测试集 =====
# 测试集特征工程（无标签列，直接处理）
test_df = add_features(test_df)

# ===== 核心修复2：仅保留双方都存在的列 =====
# 筛选训练/测试集共有的数值列（防止测试集缺失部分列）
common_numeric_cols = [col for col in original_numeric_cols if col in X.columns and col in test_df.columns]
# 筛选训练/测试集共有的类别列
common_categorical_cols = [col for col in original_categorical_cols if col in X.columns and col in test_df.columns]

# ===== 缺失值填充（用训练集中位数，仅用共有列） =====
# 训练集
X_numeric = X[common_numeric_cols].fillna(X[common_numeric_cols].median())
X_categorical = X[common_categorical_cols].fillna("Unknown")
# 测试集（复用训练集的中位数）
test_numeric = test_df[common_numeric_cols].fillna(X[common_numeric_cols].median())
test_categorical = test_df[common_categorical_cols].fillna("Unknown")

# ===================== 2. 类别特征编码 =====================
# 独热编码（仅拟合训练集，处理未知类别）
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoder.fit(X_categorical)
# 转换训练/测试集
X_encoded = encoder.transform(X_categorical)
test_encoded = encoder.transform(test_categorical)

# 合并特征（数值+编码后的类别）
X_processed = np.hstack([X_numeric.values, X_encoded])
test_processed = np.hstack([test_numeric.values, test_encoded])

# ===================== 3. K折交叉验证训练XGBoost =====================
# 初始化KFold（5折，打乱数据）
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = np.zeros(len(test_processed))
val_mae_list = []

# XGBoost核心参数（适配房价预测）
xgb_params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1,
    "random_state": 42,
    "n_jobs": -1
}

# 训练每折模型
for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed)):
    print(f"\n===== 训练第{fold+1}折 =====")
    X_train, X_val = X_processed[train_idx], X_processed[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 转换为XGBoost格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(test_processed)
    
    # 训练模型（早停防止过拟合）
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    # 计算验证集MAE
    val_pred = model.predict(dval)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_mae_list.append(val_mae)
    print(f"第{fold+1}折验证集MAE：{val_mae:.2f}")
    
    # 累加测试集预测值
    test_predictions += model.predict(dtest) / kf.n_splits

# ===================== 4. 保存预测结果 =====================
save_path = "./xgboost_predictions.csv"  # 预测结果保存到根目录(相对路径)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 构建结果文件（保留Id列）
output_df = pd.DataFrame({
    "Id": test_df["Id"],  # 确保test_df有Id列
    "SalePrice": test_predictions
})
output_df.to_csv(save_path, index=False, encoding="utf-8")

# ===================== 5. 输出最终结果 =====================
print("\n===================== 最终结果 =====================")
print(f"5折交叉验证平均MAE：{np.mean(val_mae_list):.2f}")
print(f"预测结果已保存到：{save_path}")