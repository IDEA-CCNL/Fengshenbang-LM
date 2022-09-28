import streamlit as st
from transformers import (
    AutoProcessor,
)
from transformers import HubertConfig, HubertForCTC, Wav2Vec2ForCTC, Wav2Vec2Config
import torch
import os
import subprocess
from pathlib import Path
import torchaudio
import torchaudio.functional as F
import tempfile

st.set_page_config(
    page_title="中文语音识别",  # 页面标签标题
    page_icon=":shark:",  # 页面标签图标
    layout="wide",  # 页面的布局
    initial_sidebar_state="expanded",  # 左侧的sidebar的布局方式
    # 配置菜单按钮的信息
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

#  title
st.title('Demo for ASR')

# parameter
st.sidebar.header("参数配置")
sbform = st.sidebar.form("固定参数设置")
language = sbform.selectbox('选择语言种类', ['中文', '英文'])
model_name = sbform.selectbox('选择模型种类', ['wav2vec2', 'hubert'])
sbform.form_submit_button("提交配置")

# 这里是页面中的参数配置，也是demo的主体之一
form = st.form("参数设置")


def get_model(model_name="wav2vec2"):
    if model_name == "wav2vec2":
        if language == '中文':
            model_path = "/cognitive_comp/zhuojianheng/pretrained_model/wav2vec2-base-ctc-wenet"
            ckpt = "/cognitive_comp/zhuojianheng/experiment/fengshen-wav2vec2-base-wenet-ctc-hf_pretrained_epoch14_step120000/ckpt/hf_pretrained_epoch271_step79968"
        else:
            model_path = "/cognitive_comp/zhuojianheng/pretrained_model/wav2vec2-base-ctc-libri"
            ckpt = "/cognitive_comp/zhuojianheng/experiment/fengshen-wav2vec2-base-wenet-ctc-hf_pretrained_epoch14_step120000"
        config_type = Wav2Vec2Config
        model_type = Wav2Vec2ForCTC
    elif model_name == "hubert":
        if language == '中文':
            pass
        else:
            model_path = "/cognitive_comp/zhuojianheng/pretrained_model/hubert-base-ctc/config"
        config_type = HubertConfig
        model_type = HubertForCTC

    config = config_type.from_pretrained(model_path)
    # config = HubertConfig.from_pretrained(model_path)

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    config.update(
        {
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
        }
    )
    config.vocab_size = len(processor.tokenizer.get_vocab())
    model = model_type.from_pretrained(ckpt, config=config)
    model.eval()
    device = torch.device("cuda:0")
    model.to(device)

    return model, processor


def prepare_data(processor, wav, sample_rate):
    input_features = []
    inputs = processor.feature_extractor(
        wav, sampling_rate=sample_rate,
        max_length=25000000, truncation=True
    )
    input_features.append({"input_values": inputs.input_values[0]})

    batch = processor.pad(
        input_features,
        padding=True,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )
    device = torch.device("cuda:0")
    for item in batch:
        if torch.is_tensor(batch[item]):
            batch[item] = batch[item].to(device)

    return batch


model, processor = get_model(model_name)


def asr(wav, sample_rate):
    data = prepare_data(processor, wav, sample_rate)
    labels = data.pop("labels", None)
    pred = model(**data)
    pred_logits = pred.logits
    pred_ids = torch.argmax(pred_logits, axis=-1)
    pred_str = processor.tokenizer.batch_decode(pred_ids)
    return pred_str, labels
# 模型预测结果


st.header("upload a sound file")


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[-1][1:]
    tempfilename1 = Path(tempfile.gettempdir()).joinpath(next(tempfile._get_candidate_names())+".{}".format(ext))
    byte_data = uploaded_file.getvalue()
    with open(tempfilename1, "wb") as f:
        f.write(byte_data)

    tempfilename2 = Path(tempfile.gettempdir()).joinpath(next(tempfile._get_candidate_names())+".wav")
    subprocess.run(f"ffmpeg -i {tempfilename1} -ac 1 -ar 16000 -f wav -y -loglevel 1 {tempfilename2}", shell=True)
    os.remove(tempfilename1)

    wav, sample_rate = torchaudio.backend.sox_io_backend.load(tempfilename2)
    wav = torch.squeeze(wav, 0)
    if sample_rate != 16000:
        wav = F.resample(wav, sample_rate, 16000)
        sample_rate = 16000
    # Can be used wherever a "file-like" object is accepted:
    os.remove(tempfilename2)

    with st.spinner('转换中......'):
        pred_str, labels = asr(wav, sample_rate)
        st.info("pred：{}".format(pred_str[0]))
        if labels is not None:
            st.info("label：{}".format(labels[0]))
