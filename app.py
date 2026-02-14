import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 尝试导入 skimage，如果没有则使用 OpenCV 模拟
try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def process_image(image_file, thresh, close_val, target_width, smooth_val, do_skeleton):
    # 1. 读取并解码图片
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Web端处理：为了速度，限制一下最大分辨率
    h, w = img_gray.shape
    if w > 2000:
        scale = 2000 / w
        img_gray = cv2.resize(img_gray, (0, 0), fx=scale, fy=scale)

    # 2. 放大以抗锯齿 (Upscaling)
    scale_factor = 2.0 
    h, w = img_gray.shape
    img_gray = cv2.resize(img_gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)

    # 3. 二值化
    _, binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY_INV)

    # 4. 闭合运算 (修复断点)
    real_close = int(close_val * scale_factor)
    if real_close > 0:
        k_size = real_close * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5. 骨架化
    if do_skeleton:
        binary_bool = binary > 0
        if HAS_SKIMAGE:
            skeleton = skeletonize(binary_bool)
            binary = (skeleton * 255).astype(np.uint8)
        else:
            # OpenCV 简单细化模拟
            binary = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else binary

    # 6. 膨胀 (线宽)
    real_width = int(target_width * scale_factor)
    if real_width > 0:
        k_size = real_width
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary = cv2.dilate(binary, kernel, iterations=1)

    # 7. 平滑
    real_smooth = int(smooth_val * scale_factor)
    if real_smooth > 0:
        k_blur = real_smooth * 2 + 1
        binary = cv2.GaussianBlur(binary, (k_blur, k_blur), 0)
        _, binary = cv2.threshold(binary, 100, 255, cv2.THRESH_BINARY)

    # 8. 反相
    result = cv2.bitwise_not(binary)
    return result

# === 网页界面布局 ===
st.set_page_config(page_title="3D打印线稿优化器", layout="wide")

st.title("🎨 掐丝珐琅 3D打印线稿优化器")
st.markdown("上传手绘线稿，自动转换为等宽、闭合的打印底稿。")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("参数设置")
    uploaded_file = st.file_uploader("上传图片", type=['png', 'jpg', 'jpeg'])
    
    thresh = st.slider("1. 识别阈值", 0, 255, 127, help="值越小线条越少，值越大杂噪越多")
    close_val = st.slider("2. 闭合断点力度", 0, 10, 2, help="自动连接断开的线条")
    do_skeleton = st.checkbox("3. 强制统一线宽 (骨架化)", value=True)
    target_width = st.slider("4. 最终线宽 (像素)", 1, 30, 6)
    smooth_val = st.slider("5. 边缘平滑度", 0, 10, 2)

with col2:
    st.header("效果预览")
    if uploaded_file is not None:
        # 每次参数变化，这里都会自动重新运行
        # 记得要在读取前重置指针，否则第二次读取为空
        uploaded_file.seek(0) 
        
        result_img = process_image(uploaded_file, thresh, close_val, target_width, smooth_val, do_skeleton)
        
        # 显示结果
        st.image(result_img, caption="处理结果", use_container_width=True)
        
        # 转换图片用于下载
        success, encoded_img = cv2.imencode('.png', result_img)
        if success:
            st.download_button(
                label="📥 下载处理后的图片",
                data=encoded_img.tobytes(),
                file_name="processed_lineart.png",
                mime="image/png"
            )
    else:
        st.info("👈 请在左侧上传图片开始处理")