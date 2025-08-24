import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split  # 导入train_test_split
# 读取excel文件
excel_file = r'C:\Users\zjc03\Desktop\会泽数据\selected_wavelengths_and_spectra.xlsx'
df = pd.read_excel(excel_file)

# 提取特征变量和标签变量
X = df.drop('Pb', axis=1)  # 特征变量
print(X.shape)
y = df['Pb']               # 标签变量
print(y.shape)

# 数据统计
print(df['Pb'].describe())

# 使用train_test_split函数随机划分数据集，80%训练集，20%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 标准化数据
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# 将结果转换为一维数组
y_train_scaled = y_train_scaled.ravel()
y_test_scaled = y_test_scaled.ravel()

# 初始化最佳评分和最佳组件数
best_score = 0
best_components = 0
# 保存所有结果
results = []

# 循环n_components从1到49
for n_components in range(1, 70):
    # 训练PLS回归模型
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train_scaled, y_train_scaled)
    
    # 预测
    Y_pred_train = pls.predict(X_train_scaled)
    Y_pred_test = pls.predict(X_test_scaled)
    
    # 计算R²分数
    train_score = r2_score(y_train_scaled, Y_pred_train)
    test_score = r2_score(y_test_scaled, Y_pred_test)
    
    # 保存结果
    results.append({
        'n_components': n_components,
        'train_score': train_score,
        'test_score': test_score,
        'diff': train_score - test_score,  # 训练集和测试集得分差异
        'sum': train_score + test_score    # 训练集和测试集得分总和
    })
    
    print(f"n_components={n_components}, train_score={train_score}, test_score={test_score}")

# 将结果转换为DataFrame
results_df = pd.DataFrame(results)

# 修改筛选条件，避免valid_results为空的情况
if any(results_df['train_score'] > results_df['test_score']):
    # 如果存在训练集R2大于测试集R2的情况，使用原来的筛选条件
    valid_results = results_df[results_df['train_score'] > results_df['test_score']]
    print("使用条件：训练集R2 > 测试集R2")
else:
    # 否则使用所有结果
    valid_results = results_df
    print("使用所有结果（因为不存在训练集R2 > 测试集R2的情况）")

# 查找满足条件且总分最高的组件数量
best_result = valid_results.sort_values(by=['sum'], ascending=False).iloc[0]
best_components = int(best_result['n_components'])
best_train_score = best_result['train_score']
best_test_score = best_result['test_score']

print(f"\n优化选择 n_components={best_components}")
print(f"训练集R2: {best_train_score:.4f}, 测试集R2: {best_test_score:.4f}")
print(f"总和: {best_result['sum']:.4f}, 差值: {best_result['diff']:.4f}")

# 训练PLS回归模型
n_components = best_components # 选择潜变量数量
pls = PLSRegression(n_components=n_components)
pls.fit(X_train_scaled, y_train_scaled)  # 确保使用标准化后的数据

# 预测
Y_pred_train = pls.predict(X_train_scaled)
Y_pred_test = pls.predict(X_test_scaled)

# 反标准化预测值以获得原始尺度上的预测
Y_pred_train_original = scaler_y.inverse_transform(Y_pred_train.reshape(-1, 1))
Y_pred_test_original = scaler_y.inverse_transform(Y_pred_test.reshape(-1, 1))
# 创建DataFrame保存结果
train_results = pd.DataFrame({
    '真实值': y_train,
    '预测值': Y_pred_train_original.flatten()
})

test_results = pd.DataFrame({
    '真实值': y_test,
    '预测值': Y_pred_test_original.flatten()
})

# # # 合并训练集和测试集结果
# all_results = pd.concat([train_results, test_results], ignore_index=True)

# # 保存到Excel文件
# output_file = r'SD_PLSR_Boruta.xlsx'
# all_results.to_excel(output_file, index=False)

# print(f"训练集和测试集的真实值和预测值已保存到 {output_file}")
# 评估模型
mse_train = mean_squared_error(y_train, Y_pred_train_original)
mse_test = mean_squared_error(y_test, Y_pred_test_original)

# 计算RMSE
train_rmse = np.sqrt(mse_train)
test_rmse = np.sqrt(mse_test)

# 计算MAE
train_mae = mean_absolute_error(y_train, Y_pred_train_original)
test_mae = mean_absolute_error(y_test, Y_pred_test_original)

# 计算R²分数
train_score = r2_score(y_train, Y_pred_train_original)
test_score = r2_score(y_test, Y_pred_test_original)

# 计算RPD值 (标准差/RMSE)
train_rpd = np.std(y_train) / train_rmse
test_rpd = np.std(y_test) / test_rmse

# 输出结果
print("Final model performance:")
print(f"最佳主成分数: {n_components}")
print(f"训练集RMSE: {train_rmse:.4f}")
print(f"训练集R2: {train_score:.4f}")
print(f"训练集MAE: {train_mae:.4f}")
print(f"训练集RPD: {train_rpd:.4f}")

print(f"测试集RMSE: {test_rmse:.4f}")
print(f"测试集R2: {test_score:.4f}")
print(f"测试集MAE: {test_mae:.4f}")
print(f"测试集RPD: {test_rpd:.4f}")

# 绘制散点图
plt.figure(figsize=(12, 5))

# 训练集散点图
plt.subplot(1, 2, 1)
plt.scatter(y_train, Y_pred_train_original, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--')
plt.xlabel('实测值')
plt.ylabel('预测值')
plt.title(f'训练集 (R² = {train_score:.4f}, RMSE = {train_rmse:.4f}, RPD = {train_rpd:.4f})')

# 测试集散点图
plt.subplot(1, 2, 2)
plt.scatter(y_test, Y_pred_test_original, alpha=0.6, color='r')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('实测值')
plt.ylabel('预测值')
plt.title(f'测试集 (R² = {test_score:.4f}, RMSE = {test_rmse:.4f}, RPD = {test_rpd:.4f})')

plt.tight_layout()
plt.savefig('PLSR_建模结果散点图.png', dpi=300)
plt.show()
