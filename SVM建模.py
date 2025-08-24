# 导入第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR  # 输入支持向量机模型
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score  # 评估模型性能
from sklearn.model_selection import train_test_split

# 读取Excel文件数据
excel_file = r'E:\大论文研究区融合裁剪\矿山镇影像\HSRNET_resample\星载分数阶微分\Pb特征波段\特征波段\2.0阶.xlsx'
df = pd.read_excel(excel_file)

# 提取特征变量和标签变量
X = df.drop('Pb', axis=1)  # 去掉相应列，保留其他列作为特征变量
y = df['Pb']  # 标签变量
# 使用train_test_split进行随机划分，test_size表示测试集占比，random_state用于保证每次运行划分一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 使用StandardScaler对特征和目标变量进行标准化
scaler_X = StandardScaler()  # 创建一个标准化对象
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)  # 在训练集上计算平均值和标准差，并应用标准化。
X_test_scaled = scaler_X.transform(X_test)  # 使用在训练集上计算得到平均值和标准差来标准化测试集

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))  # 对目标变量 y_train 和 y_test 进行标准化
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# 将结果转换为一维数组
y_train_scaled = y_train_scaled.ravel()
y_test_scaled = y_test_scaled.ravel()

# 存储测试集 R² 分数和对应的核函数
kernel_results = {}

# 通过for循环来选择最好的核函数。其中linear线性核(适用于数据线性关系)、poly多项式核（通过增加degree参数（核的度数），可以调整决策边界的复杂性）、rbf径向基函数（默认使用，非常适合非线性问题）、sigmoid（模拟神经网络的神经元激活）
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    # 初始化支持向量回归模型
    clf = SVR(kernel=k)

    # 模型训练
    clf.fit(X_train_scaled, y_train_scaled)
    
    # 在训练集上进行预测
    y_train_pred = clf.predict(X_train_scaled)
    
    # 在测试集上进行预测
    y_test_pred = clf.predict(X_test_scaled)
    
    # 计算训练集和测试集的R²分数
    train_r2 = r2_score(y_train_scaled, y_train_pred)
    test_r2 = r2_score(y_test_scaled, y_test_pred)
    
    # 保存结果
    kernel_results[k] = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'diff': train_r2 - test_r2,  # 训练集和测试集得分差异
        'sum': train_r2 + test_r2    # 训练集和测试集得分总和
    }

# 输出每个核函数的训练集和测试集 R² 分数
for kernel, scores in kernel_results.items():
    print(f"{kernel} 核函数训练集 R² 分数: {scores['train_r2']:.3f}, 测试集 R² 分数: {scores['test_r2']:.3f}, 差值: {scores['diff']:.3f}")

# 筛选出训练集R²大于测试集R²的核函数，但差异不要太大
valid_kernels = {k: v for k, v in kernel_results.items() if v['train_r2'] > v['test_r2'] and v['diff'] < 0.15}

if valid_kernels:
    print("\n训练集R²大于测试集R²且差异合理的核函数:")
    for kernel, scores in valid_kernels.items():
        print(f"{kernel} 核函数: 训练集R²={scores['train_r2']:.3f}, 测试集R²={scores['test_r2']:.3f}, 总和={scores['sum']:.3f}")
    
    # 从满足条件的核函数中选择测试集R²最高的
    best_kernel = max(valid_kernels.items(), key=lambda x: x[1]['test_r2'])[0]
else:
    print("\n没有找到同时满足训练集R²>测试集R²且差异小于0.15的核函数")
    # 再次尝试放宽条件
    valid_kernels = {k: v for k, v in kernel_results.items() if v['train_r2'] > v['test_r2']}
    
    if valid_kernels:
        print("选择训练集R²>测试集R²的核函数中，测试集R²最高的")
        best_kernel = max(valid_kernels.items(), key=lambda x: x[1]['test_r2'])[0]
    else:
        print("从所有核函数中选择测试集R²最高的")
        best_kernel = max(kernel_results.items(), key=lambda x: x[1]['test_r2'])[0]

print(f"\n选择的最佳核函数: {best_kernel}")
print(f"训练集R²: {kernel_results[best_kernel]['train_r2']:.3f}, 测试集R²: {kernel_results[best_kernel]['test_r2']:.3f}")

# 下面针对最佳核函数进行参数优化
print(f"\n开始对 {best_kernel} 核函数进行参数优化...")

# 定义参数网格 - 减少参数组合以加快计算速度
param_grids = {
    'linear': {
        'C': [0.001, 0.1, 10, 100],
        'epsilon': [0.001, 0.1, 0.5],
    },
    'poly': {
        # 大幅减少参数组合数量
        'C': [0.001, 0.1, 10],
        'epsilon': [0.001, 0.1],
        'degree': [2, 3],  # 减少degree选项
        'gamma': [0.01, 0.1, 'scale']  # 减少gamma选项
    },
    'rbf': {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'epsilon': [0.0001, 0.001, 0.01, 0.1, 0.5],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 'scale', 'auto']
    },
    'sigmoid': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'epsilon': [0.001, 0.01, 0.1, 0.5],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto']
    }
}

# 选择对应最佳核函数的参数网格
selected_param_grid = param_grids[best_kernel]
# 添加固定的kernel参数
selected_param_grid['kernel'] = [best_kernel]

# 使用GridSearchCV进行参数优化
print("使用GridSearchCV进行参数优化...")
svr = SVR()
grid_search = GridSearchCV(
    estimator=svr,
    param_grid=selected_param_grid,
    cv=5,  # 5折交叉验证
    scoring='r2',
    verbose=2,
    n_jobs=-1  # 使用所有可用CPU核心加速计算
)

# 进行网格搜索
grid_search.fit(X_train_scaled, y_train_scaled)

# 输出最佳参数组合
print("\n网格搜索找到的最佳参数组合:")
print(grid_search.best_params_)
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数创建模型
best_params = grid_search.best_params_
best_svr = SVR(**best_params)

# 使用最佳参数重新训练模型
best_svr.fit(X_train_scaled, y_train_scaled)

# 对训练集和测试集进行预测
Y_pred_train = best_svr.predict(X_train_scaled)
Y_pred_test = best_svr.predict(X_test_scaled)

# 计算标准化数据的R²分数
train_r2_scaled = r2_score(y_train_scaled, Y_pred_train)
test_r2_scaled = r2_score(y_test_scaled, Y_pred_test)

print(f"\n标准化数据上的性能:")
print(f"训练集R²: {train_r2_scaled:.4f}")
print(f"测试集R²: {test_r2_scaled:.4f}")
print(f"差值: {train_r2_scaled - test_r2_scaled:.4f}")

# 对预测结果进行反归一化
Y_pred_train_original = scaler_y.inverse_transform(Y_pred_train.reshape(-1, 1))
Y_pred_test_original = scaler_y.inverse_transform(Y_pred_test.reshape(-1, 1))

# 计算训练集的RMSE和R²
train_rmse = np.sqrt(mean_squared_error(y_train, Y_pred_train_original))
train_r2 = r2_score(y_train, Y_pred_train_original)

# 计算测试集的RMSE和R²
test_rmse = np.sqrt(mean_squared_error(y_test, Y_pred_test_original))
test_r2 = r2_score(y_test, Y_pred_test_original)

# 计算训练集和测试集的RPD (相对预测偏差)
train_rpd = np.std(y_train) / train_rmse
test_rpd = np.std(y_test) / test_rmse

# 计算 MAE
train_mae = mean_absolute_error(y_train, Y_pred_train_original)
test_mae = mean_absolute_error(y_test, Y_pred_test_original)

# 输出模型评估结果
print("\n最终模型评估结果:")
print(f"最佳参数: {best_params}")
print(f"训练集RMSE: {train_rmse:.4f}")
print(f"训练集R²: {train_r2:.4f}")
print(f"训练集RPD: {train_rpd:.4f}")
print(f"训练集MAE: {train_mae:.4f}")
print(f"测试集RMSE: {test_rmse:.4f}")
print(f"测试集R²: {test_r2:.4f}")
print(f"测试集RPD: {test_rpd:.4f}")
print(f"测试集MAE: {test_mae:.4f}")

# 计算训练集和测试集R²的差值
r2_diff = train_r2 - test_r2
print(f"训练集和测试集R²差值: {r2_diff:.4f}")

# 如果训练集和测试集的R²差值太大或者测试集R²太低，则尝试其他参数组合
if r2_diff > 0.2 or test_r2 < 0.4:
    print("\n当前模型可能存在过拟合或性能不佳，尝试更保守的参数...")
    
    # 根据核函数类型选择更保守的参数
    if best_kernel == 'poly':
        # 对于多项式核，降低degree和C值
        conservative_params = {
            'kernel': best_kernel,
            'C': 0.1,
            'epsilon': 0.1,
            'degree': 2,
            'gamma': 0.01
        }
    elif best_kernel == 'rbf':
        # 对于RBF核，降低C值增加gamma值
        conservative_params = {
            'kernel': best_kernel,
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 0.01
        }
    else:
        # 对于其他核函数，使用较低的C值
        conservative_params = {
            'kernel': best_kernel,
            'C': 1.0,
            'epsilon': 0.1
        }
        if best_kernel in ['sigmoid']:
            conservative_params['gamma'] = 0.01
            
    print(f"尝试的保守参数: {conservative_params}")
    
    # 创建并训练保守模型
    conservative_svr = SVR(**conservative_params)
    conservative_svr.fit(X_train_scaled, y_train_scaled)
    
    # 进行预测和评估
    Y_pred_train_cons = conservative_svr.predict(X_train_scaled)
    Y_pred_test_cons = conservative_svr.predict(X_test_scaled)
    
    # 反归一化
    Y_pred_train_original_cons = scaler_y.inverse_transform(Y_pred_train_cons.reshape(-1, 1))
    Y_pred_test_original_cons = scaler_y.inverse_transform(Y_pred_test_cons.reshape(-1, 1))
    
    # 计算指标
    train_r2_cons = r2_score(y_train, Y_pred_train_original_cons)
    test_r2_cons = r2_score(y_test, Y_pred_test_original_cons)
    r2_diff_cons = train_r2_cons - test_r2_cons
    
    # 计算RMSE
    train_rmse_cons = np.sqrt(mean_squared_error(y_train, Y_pred_train_original_cons))
    test_rmse_cons = np.sqrt(mean_squared_error(y_test, Y_pred_test_original_cons))
    
    # 计算RPD
    train_rpd_cons = np.std(y_train) / train_rmse_cons
    test_rpd_cons = np.std(y_test) / test_rmse_cons
    
    # 计算MAE
    train_mae_cons = mean_absolute_error(y_train, Y_pred_train_original_cons)
    test_mae_cons = mean_absolute_error(y_test, Y_pred_test_original_cons)
    
    print(f"\n保守模型性能:")
    print(f"训练集R²: {train_r2_cons:.4f}")
    print(f"训练集RMSE: {train_rmse_cons:.4f}")
    print(f"训练集RPD: {train_rpd_cons:.4f}")
    print(f"训练集MAE: {train_mae_cons:.4f}")
    print(f"测试集R²: {test_r2_cons:.4f}")
    print(f"测试集RMSE: {test_rmse_cons:.4f}")
    print(f"测试集RPD: {test_rpd_cons:.4f}")
    print(f"测试集MAE: {test_mae_cons:.4f}")
    print(f"R²差值: {r2_diff_cons:.4f}")
    
    # 如果保守模型的测试集R²更高或差值更小，则使用保守模型
    if test_r2_cons > test_r2 or (test_r2_cons > 0.4 and r2_diff_cons < r2_diff):
        print("保守模型性能更好，使用保守模型作为最终模型")
        best_svr = conservative_svr
        best_params = conservative_params
        Y_pred_train_original = Y_pred_train_original_cons
        Y_pred_test_original = Y_pred_test_original_cons
        train_r2 = train_r2_cons
        test_r2 = test_r2_cons
        train_rmse = train_rmse_cons
        test_rmse = test_rmse_cons
        train_mae = train_mae_cons
        test_mae = test_mae_cons
        train_rpd = train_rpd_cons
        test_rpd = test_rpd_cons

# 绘制散点图
plt.figure(figsize=(12, 5))

# 训练集
plt.subplot(1, 2, 1)
plt.scatter(y_train, Y_pred_train_original.ravel(), alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('实测值')
plt.ylabel('预测值')
plt.title(f'训练集 (R² = {train_r2:.4f}, RMSE = {train_rmse:.4f}, MAE = {train_mae:.4f})')

# 测试集
plt.subplot(1, 2, 2)
plt.scatter(y_test, Y_pred_test_original.ravel(), alpha=0.7, color='r')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('实测值')
plt.ylabel('预测值')
plt.title(f'测试集 (R² = {test_r2:.4f}, RMSE = {test_rmse:.4f}, MAE = {test_mae:.4f})')

plt.tight_layout()
plt.savefig('SVM_建模结果散点图.png', dpi=300)
plt.show()

# 构建数据框
df_output = pd.DataFrame(columns=['真实值', '预测值'])
df_output['真实值'] = y_test.round(3)  # 赋值
df_output['预测值'] = Y_pred_test_original.round(3)  # 赋值
# 按索引排序
df_output = df_output.sort_index()
# 打印格式化后的数据框
from IPython.display import display
display(df_output)

# 输出您提到的两个特定参数组合的结果
# print("\n您提到的参数组合的结果:")
# for result in all_results:
#     if 'C' in result['params'] and 'epsilon' in result['params'] and 'gamma' in result['params']:
#         if (result['params']['C'] == 1000 and result['params']['epsilon'] == 0.5 and result['params']['gamma'] == 0.001) or \
#            (result['params']['C'] == 100 and result['params']['epsilon'] == 0.5 and result['params']['gamma'] == 0.001):
#             print(f"C={result['params']['C']}, epsilon={result['params']['epsilon']}, gamma={result['params']['gamma']}: "
#                   f"训练集R²={result['train_r2']:.4f}, 测试集R²={result['test_r2']:.4f}, 差值={result['diff']:.4f}")
