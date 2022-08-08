import requests
import langid
import streamlit as st
from translate import baiduTranslatorMedical
from translate import baiduTranslator

langid.set_languages(['en', 'zh'])
lang_dic = {'zh': 'en', 'en': 'zh'}

st.set_page_config(
    page_title="余元医疗问答",
    page_icon=":shark:",
    #  layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.title('Demo for MedicalQA')


st.sidebar.header("参数配置")
sbform = st.sidebar.form("固定参数设置")
n_sample = sbform.slider("设置返回条数", min_value=1, max_value=10, value=3)
text_length = sbform.slider('生成长度:', min_value=32, max_value=512, value=64, step=32)
text_level = sbform.slider('文本多样性:', min_value=0.1, max_value=1.0, value=0.9, step=0.1)
model_id = sbform.number_input('选择模型号:', min_value=0, max_value=13, value=13, step=1)
trans = sbform.selectbox('选择翻译内核', ['百度通用', '医疗生物'])
sbform.form_submit_button("配置")


form = st.form("参数设置")
input_text = form.text_input('请输入你的问题:', value='', placeholder='例如：糖尿病的症状有哪些？')
if trans == '百度通用':
    translator = 'baidu_common'
else:
    translator = 'baidu'
if input_text:
    lang = langid.classify(input_text)[0]
    if translator == 'baidu':
        st.write('**你的问题是:**', baiduTranslatorMedical(input_text, src=lang, dest=lang_dic[lang]).text)
    else:
        st.write('**你的问题是:**', baiduTranslator(input_text, src=lang, dest=lang_dic[lang]).text)

form.form_submit_button("提交")

# @st.cache(suppress_st_warning=True)


def generate_qa(input_text, n_sample, model_id='7', length=64, translator='baidu', level=0.7):
    # st.write('调用了generate函数')
    URL = 'http://192.168.190.63:6605/qa'
    data = {"text": input_text, "n_sample": n_sample, "model_id": model_id,
            "length": length, 'translator': translator, 'level': level}
    r = requests.get(URL, params=data)
    return r.text
# my_bar = st.progress(80)


with st.spinner('老夫正在思考中🤔...'):
    if input_text:
        results = generate_qa(input_text, n_sample, model_id=str(model_id),
                              translator=translator, length=text_length, level=text_level)
        for idx, item in enumerate(eval(results), start=1):
            st.markdown(f"""
            **候选回答「{idx}」:**\n
            """)
            st.info('中文：%s' % item['fy_next_sentence'])
            st.info('英文：%s' % item['next_sentence'])
