import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_excel(r"C:\Users\zjc03\Desktop\FD.xlsx")

# 准备特征和目标变量
X = df.drop(columns=['Pb'])
y = df['Pb']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义网格搜索参数
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 5, 6, 7, 8, 9, 10, None],
    'min_samples_split': [2, 3, 4, 5, 6,10],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10]
}

# 创建随机森林回归器
rf = RandomForestRegressor(random_state=42)

# 创建网格搜索对象
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# 执行网格搜索
print("开始网格搜索最优参数...")
grid_search.fit(X_train, y_train)

# 输出最优参数
print("\n最优参数:", grid_search.best_params_)
print("最优得分:", -grid_search.best_score_)

# 使用最优参数创建模型
best_rf = RandomForestRegressor(**grid_search.best_params_, random_state=42)
best_rf.fit(X_train, y_train)

# 训练集预测
y_train_pred = best_rf.predict(X_train)
# 测试集预测
y_test_pred = best_rf.predict(X_test)

# 计算RPD（相对预测偏差）
def calculate_rpd(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std = np.std(y_true)
    return std / rmse

# 训练集评估
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rpd = calculate_rpd(y_train, y_train_pred)

# 测试集评估
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rpd = calculate_rpd(y_test, y_test_pred)

# 打印评估结果
print("\n训练集评估结果:")
print(f"R² 分数: {train_r2:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"MAE: {train_mae:.4f}")
print(f"RPD: {train_rpd:.4f}")

print("\n测试集评估结果:")
print(f"R² 分数: {test_r2:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"RPD: {test_rpd:.4f}")

# 特征重要性分析
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': best_rf.feature_importances_
})
feature_importance = feature_importance.sort_values('重要性', ascending=False)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['特征'], feature_importance['重要性'])
plt.xticks(rotation=45, ha='right')
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.show()

# 绘制预测值与真实值对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('随机森林回归预测结果对比')
plt.tight_layout()
plt.show() 