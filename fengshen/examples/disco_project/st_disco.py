# from disco_huge import Diffuser
# from utils import *
from disco import Diffuser
import streamlit as st
from io import BytesIO
from PIL import Image
from disco import steps


@st.cache(show_spinner=False, allow_output_mutation=True)   # 加装饰器， 只加载一次。
class ST_Diffuser(Diffuser):
    def __init__(self, custom_path):
        super().__init__(custom_path)


if __name__ == '__main__':
    dd = ST_Diffuser(custom_path="IDEA-CCNL/Taiyi-Diffusion-532M-Nature")  # 初始化
    form = st.form("参数设置")
    input_text = form.text_input('输入文本生成图像:', value='', placeholder='你想象的一个画面')
    form.form_submit_button("提交")
    uploaded_file = st.file_uploader("上传初始化图片（可选）", type=["jpg", "png", "jpeg"])

    text_scale_norm = st.sidebar.slider('文本强度', 0.1, 1.0, 0.5, step=0.1)
    text_scale = int(text_scale_norm * 10000)
    res_skip_steps = st.sidebar.slider('加噪强度', 0.1, 1.0, 0.9, step=0.1)
    skip_steps = int(steps - round(res_skip_steps * steps))
    width = st.sidebar.slider('宽度', 384, 1024, 512, step=64)
    heigth = st.sidebar.slider('高度', 384, 1024, 512, step=64)

    with st.spinner('正在生成中...'):
        capture_img = None
        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            # 将字节数据转化成字节流
            bytes_data = BytesIO(bytes_data)
            # Image.open()可以读字节流
            capture_img = Image.open(bytes_data).convert('RGB').resize((width, heigth))

            image_status = st.empty()
            image_status.image(capture_img, use_column_width=True)
        else:
            image_status = st.empty()

        if input_text:
            # global text_prompts
            input_text_prompts = [input_text]
            image = dd.generate(input_text_prompts,
                                capture_img,
                                clip_guidance_scale=text_scale,
                                skip_steps=skip_steps,
                                st_dynamic_image=image_status,
                                init_scale=None,
                                side_x=width,
                                side_y=heigth)   # 最终结果。实时显示修改generate里面的内容。
            image_status.image(image, use_column_width=True)
