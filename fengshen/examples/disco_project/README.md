# Chinese Warp For Disco Diffusion
- This is a chinese version disco diffusion. We train a Chinese CLIP [IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese) and utilize it to guide the diffusion process. 
- This code is modified from https://github.com/alembics/disco-diffusion
- streamlit demo is supported.
- the checkpoint has been upload to hugging face.
## Usage

- setup the environment by`pip install -r requirement.txt` or install the lack package directly
### Run Directly 
```
python disco.py
```

### Streamlit Setup
```
streamlit run st_disco.py
# --server.port=xxxx --server.address=xxxx
```
