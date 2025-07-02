import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 数据上传和预处理
st.title("XGBoost模型与SHAP分析")
st.subheader("步骤1：上传CSV文件")

uploaded_file = st.file_uploader("选择一个CSV文件", type=["csv"])

if uploaded_file is not None:
    # 加载数据
    df = pd.read_csv(uploaded_file)
    st.write("数据预览：", df.head())

    # 自动识别因变量Y和自变量X
    st.subheader("步骤2：选择自变量和因变量")
    target_column = st.selectbox("选择因变量（Y）", df.columns)
    feature_columns = st.multiselect("选择自变量（X）", df.columns.tolist(), default=df.columns.tolist())

    if target_column and feature_columns:
        # 划分数据集
        X = df[feature_columns]
        y = df[target_column]
        
        # 数据集拆分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 检查缺失值
        if np.any(np.isnan(y_test)) or np.any(np.isnan(y_pred)):
            st.error("数据包含缺失值（NaN），请检查数据！")
        else:
            # 确保数据为浮动类型
            y_test = y_test.astype(float)
            y_pred = y_pred.astype(float)
            
            # XGBoost模型训练
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
            model.fit(X_train, y_train)
            
            # 预测与评估
            y_pred = model.predict(X_test)
            
            # 计算 MSE 和 RMSE
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            # 计算 R²
            r2 = r2_score(y_test, y_pred)
            
            st.subheader(f"模型结果")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"R²: {r2:.4f}")
            
            # 特征重要性
            st.subheader("特征重要性")
            importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': importance
            }).sort_values(by="Importance", ascending=False)
            st.write(importance_df)

            # 绘制特征重要性图
            fig, ax = plt.subplots()
            ax.barh(importance_df['Feature'], importance_df['Importance'])
            st.pyplot(fig)

            # SHAP值分析
            st.subheader("步骤4：SHAP分析")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)

            # SHAP Summary Plot
            st.write("SHAP Summary Plot")
            shap.summary_plot(shap_values, X_train)
            
            # SHAP Force Plot
            st.write("选择单个样本查看SHAP Force Plot")
            sample_index = st.slider("选择样本索引", 0, len(X_train) - 1, 0)
            shap.initjs()
            st.components.v1.html(
                shap.force_plot(explainer.expected_value, shap_values[sample_index, :], X_train.iloc[sample_index, :], matplotlib=True), 
                height=500
            )

    else:
        st.warning("请确保选择了因变量和自变量。")
else:
    st.warning("请上传CSV文件。")
