import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps

# 尝试导入 skimage，如果没有则使用 OpenCV 模拟
try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def safe_load_image(uploaded_file):
    """
    安全加载图片：处理透明背景，确保是白底黑线
    """
    # 1. 使用 PIL 读取，它对格式支持最好
    image = Image.open(uploaded_file)

    # 2. 处理透明通道 (Alpha Channel)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # 创建一个白色的背景
        alpha = image.convert('RGBA').split()[-1]
        bg = Image.new("RGB", image.size, (255, 255, 255))
        # 将原图粘贴到白底上，使用 Alpha 通道作为掩码
        bg.paste(image, mask=alpha)
        image = bg
    else:
        # 如果不是透明的，直接转 RGB 确保兼容性
        image = image.convert("RGB")

    # 3. 增加白边 (Padding) - 解决多余边框线问题
    # 在四周增加 20 像素的白边
    image = ImageOps.expand(image, border=20, fill='white')

    return image

def process_image(pil_image, thresh, close_val, target_width, smooth_val, do_skeleton):
    # 将 PIL 图片转换为 OpenCV 格式 (灰度)
    img_np = np.array(pil_image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Web端处理：为了速度，限制最大分辨率
    h, w = img_gray.shape
    if w > 2000:
        scale = 2000 / w
        img_gray = cv2.resize(img_gray, (0, 0), fx=scale, fy=scale)

    # 1. 放大以抗锯齿 (Upscaling)
    scale_factor = 2.0 
    h, w = img_gray.shape
    img_gray = cv2.resize(img_gray, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)

    # 2. 二值化 (Threshold) - 确保黑线白底
    # 这一步非常关键：使用 THRESH_BINARY_INV，前提是背景必须是亮的
    _, binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY_INV)

    # 3. 闭合运算 (修复断点)
    real_close = int(close_val * scale_factor)
    if real_close > 0:
        k_size = real_close * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. 骨架化 (Skeletonize)
    if do_skeleton:
        binary_bool = binary > 0
        if HAS_SKIMAGE:
            skeleton = skeletonize(binary_bool)
            binary = (skeleton * 255).astype(np.uint8)
        else:
            binary = cv2.ximgproc.thinning(binary) if hasattr(cv2, 'ximgproc') else binary

    # 5. 膨胀 (线宽)
    real_width = int(target_width * scale_factor)
    if real_width > 0:
        k_size = real_width
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary = cv2.dilate(binary, kernel, iterations=1)

    # 6. 平滑 (Anti-aliasing)
    real_smooth = int(smooth_val * scale_factor)
    if real_smooth > 0:
        k_blur = real_smooth * 2 + 1
        binary = cv2.GaussianBlur(binary, (k_blur, k_blur), 0)
        # 软阈值切割
        _, binary = cv2.threshold(binary, 110, 255, cv2.THRESH_BINARY)

    # 7. 反相回白底黑线
    result = cv2.bitwise_not(binary)
    
    # 8. (可选) 去除之前加的白边，或者保留也行，3D打印通常不介意白边
    # 这里我们保留白边，比较安全
    
    return result

# === 网页界面布局 ===
st.set_page_config(page_title="线稿优化神器 (增强版)", layout="wide")

st.title("🛠️ 增强版：线稿优化器")
st.markdown("已修复透明PNG变全黑、边缘出现多余线条的问题。")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. 上传")
    uploaded_file = st.file_uploader("选择图片 (支持 PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    
    st.header("2. 设置")
    thresh = st.slider("识别阈值 (杂点过滤)", 0, 255, 140, help="如果出现全黑，请降低此数值；如果线条断裂，请提高此数值")
    close_val = st.slider("闭合断点力度", 0, 10, 2)
    target_width = st.slider("最终线宽 (像素)", 1, 40, 8)
    smooth_val = st.slider("边缘平滑度", 0, 10, 3)
    do_skeleton = st.checkbox("强制统一线宽 (骨架化)", value=True)

with col2:
    st.header("3. 结果")
    if uploaded_file is not None:
        try:
            # 使用新的安全加载函数
            pil_img = safe_load_image(uploaded_file)
            
            # 处理图片
            result_img = process_image(pil_img, thresh, close_val, target_width, smooth_val, do_skeleton)
            
            # 显示
            st.image(result_img, caption="优化后的底稿", use_container_width=True)
            
            # 下载
            success, encoded_img = cv2.imencode('.png', result_img)
            if success:
                st.download_button(
                    label="📥 下载优化后的 PNG",
                    data=encoded_img.tobytes(),
                    file_name="optimized_lineart.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"处理出错: {e}")
            st.info("请尝试调整'识别阈值'或更换一张图片。")
    else:
        st.info("请在左侧上传图片。")
