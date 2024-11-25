from cuml.svm import SVC
from utils_data_preprocessing import create_file_name, load_and_split_data, remove_highly_correlated_features, standardize_features, convert_to_numpy
from utils_training import perform_grid_search_cv, evaluate_and_save_results

# 创建文件名
file_name_without_extension = create_file_name(file_path=__file__)

# 读取数据并分割为训练集和测试集
X_train, X_test, Y_train, Y_test = load_and_split_data('data.csv')

# 移除高度相关的特征
X_train, X_test = remove_highly_correlated_features(X_train, X_test)

# 特征标准化
X_train, X_test = standardize_features(X_train, X_test)

# 将数据从cudf或cupy转换为numpy数组
X_train_np, X_test_np, Y_train_np, Y_test_np = convert_to_numpy(X_train, X_test, Y_train, Y_test)

if __name__ == '__main__':
    parameters = {
        'kernel': ['linear', 'rbf'],
        'C': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 250, 500, 1000],
        'gamma': [0.001, 0.0005, 0.0001]
    }
    model = SVC()

    # 执行网格搜索和模型训练
    best_params, best_model = perform_grid_search_cv(model, parameters, X_train_np, Y_train_np, cv=10)

    # 评估模型并保存结果
    evaluate_and_save_results(best_model, X_test_np, Y_test_np, file_name_without_extension, best_params=best_params,
                              use_decision_function=True)