import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- 配置页面 ---
st.set_page_config(
    page_title="B站番剧播放量预测器",  # 页面标题
    page_icon="📊",                 # 页面图标
    layout="wide",                 # 页面布局 ('centered' 或 'wide')
    initial_sidebar_state="expanded" # 侧边栏状态 ('auto', 'expanded', 'collapsed')
)

# --- 加载模型 ---
MODEL_PATH = 'random_forest_model.pkl' # 模型文件路径

@st.cache_resource # 使用缓存加载模型，提高性能
def load_model(path):
    """加载pickle格式的模型文件"""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        print("模型加载成功") # 控制台输出加载成功信息
        return model
    except FileNotFoundError:
        st.error(f"错误: 模型文件 {path} 未找到。请确保模型文件在同一目录下。") # 在网页上显示错误信息
        print(f"错误: 文件 {path} 未找到") # 控制台输出错误信息
        return None
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}") # 在网页上显示通用错误信息
        print(f"加载模型时出错: {str(e)}") # 控制台输出错误信息
        return None

model = load_model(MODEL_PATH) # 加载模型

# --- 定义特征映射（用于用户界面显示） ---
type_options = {
    1: '少儿教育',
    2: '幻想冒险',
    3: '情感生活',
    4: '悬疑惊悚',
    5: '文艺历史'
}

adapt_options = {
    0: '否',
    1: '是'
}

time_options = {
    1: '假期档 (1/2/7/8月)',
    0: '非假期档'
}

exclusive_options = {
    0: '否',
    1: '是'
}

origin_options = {
    1: '日本',
    2: '美国',
    3: '中国'
}

# --- Streamlit 界面 ---
st.title("📊 B站番剧总播放量预测器") # 应用主标题
st.markdown("使用训练好的随机森林模型，根据输入的番剧特征预测其总播放量（万）") # 应用描述

# --- 输入区域 ---
st.sidebar.header("⚙️ 请输入番剧特征") # 侧边栏标题

# 使用列布局优化输入区域
col1, col2 = st.sidebar.columns(2)

with col1:
    # --- 分类特征输入 (使用下拉选择框 Selectbox) ---
    type_label = st.selectbox("动画类型:", options=list(type_options.keys()), format_func=lambda x: f"{type_options[x]} ({x})")
    adapt_label = st.selectbox("是否改编:", options=list(adapt_options.keys()), format_func=lambda x: f"{adapt_options[x]} ({x})")
    time_label = st.selectbox("开播时间:", options=list(time_options.keys()), format_func=lambda x: f"{time_options[x]} ({x})")
    exclusive_label = st.selectbox("是否独家:", options=list(exclusive_options.keys()), format_func=lambda x: f"{exclusive_options[x]} ({x})")
    origin_label = st.selectbox("产地:", options=list(origin_options.keys()), format_func=lambda x: f"{origin_options[x]} ({x})")
    episodes = st.number_input("集数:", min_value=1, value=12, step=1) # 数值型：集数

with col2:
    # --- 数值特征输入 (使用数字输入框 Number Input) ---
    likes = st.number_input("点赞数(PV):", min_value=0, value=460000, step=1000)
    coins = st.number_input("投币数(PV):", min_value=0, value=218000, step=1000)
    favorites = st.number_input("收藏数(PV):", min_value=0, value=53000, step=1000)
    shares = st.number_input("分享数(PV):", min_value=0, value=19000, step=1000)

st.sidebar.subheader("📝 主题模型权重 (Topic Weights)") # 主题权重的小标题
topic_cols = st.sidebar.columns(5) # 为5个Topic权重创建5列
with topic_cols[0]:
    topic0 = st.number_input("Topic 0:", min_value=0.0, max_value=1.0, value=0.41, step=0.01, format="%.2f")
with topic_cols[1]:
    topic1 = st.number_input("Topic 1:", min_value=0.0, max_value=1.0, value=0.14, step=0.01, format="%.2f")
with topic_cols[2]:
    topic2 = st.number_input("Topic 2:", min_value=0.0, max_value=1.0, value=0.11, step=0.01, format="%.2f")
with topic_cols[3]:
    topic3 = st.number_input("Topic 3:", min_value=0.0, max_value=1.0, value=0.21, step=0.01, format="%.2f")
with topic_cols[4]:
    topic4 = st.number_input("Topic 4:", min_value=0.0, max_value=1.0, value=0.12, step=0.01, format="%.2f")

# 检查Topic权重和是否接近1 (可选，给用户提示)
topic_sum = topic0 + topic1 + topic2 + topic3 + topic4
if not np.isclose(topic_sum, 1.0, atol=0.05): # 允许一点误差
    st.sidebar.warning(f"注意: Topic权重的总和 ({topic_sum:.2f}) 不等于 1。请检查输入")

# --- 预测按钮和结果显示 ---
if st.sidebar.button("🚀 预测总播放量", type="primary"): # 添加预测按钮
    if model is not None: # 确保模型已加载
        # 1. 准备输入数据
        input_data = {
            '类型': type_label,
            '是否改编': adapt_label,
            '开播时间': time_label,
            '是否独家': exclusive_label,
            '产地': origin_label,
            '集数': episodes,
            '点赞数（PV）': likes,
            '投币数（PV）': coins,
            '收藏数（PV）': favorites,
            '分享数（PV）': shares,
            'Topic 0': topic0,
            'Topic 1': topic1,
            'Topic 2': topic2,
            'Topic 3': topic3,
            'Topic 4': topic4
        }

        # 2. 转换为DataFrame，并确保特征顺序正确
        input_df = pd.DataFrame([input_data])
        required_features = ['类型','是否改编','开播时间','是否独家','产地','集数','点赞数（PV）',
                           '投币数（PV）','收藏数（PV）','分享数（PV）','Topic 0','Topic 1',
                           'Topic 2','Topic 3','Topic 4'] # 模型训练时的特征顺序
        try:
            input_df = input_df[required_features] # 按照训练时的顺序排列特征

            # 3. 进行预测
            prediction = model.predict(input_df)
            predicted_value = prediction[0] # 获取预测结果

            # 4. 显示预测结果
            st.subheader("📈 预测结果") # 结果区域标题
            st.metric(label="预测总播放量", value=f"{predicted_value:.2f} 万") # 使用metric显示结果，更美观
            st.success("预测完成！") # 显示成功信息

            # 添加一些解释性文本
            st.markdown("""
            ---
            **说明:**
            *    🔥 该预测结果基于输入的特征和训练好的随机森林模型

            """)

        except KeyError as e:
            st.error(f"输入数据准备错误: 缺少特征 {str(e)}。请检查代码或输入。") # 处理特征名称不匹配错误
        except Exception as e:
            st.error(f"预测过程中发生错误: {str(e)}") # 处理其他预测错误

    else:
        st.error("模型未能成功加载，无法进行预测") # 如果模型未加载，提示用户

# --- 页脚信息  ---
st.sidebar.markdown("---")
st.sidebar.info("💬 模型: 随机森林回归 | 数据来源: Bilibili")