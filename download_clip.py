import os
 
# 下载模型
os.system('huggingface-cli download --token ********************************** '
'--resume-download --local-dir-use-symlinks False openai/clip-vit-large-patch14 '
'--local-dir /data01/yihan/brain-diffuser/versatile_diffusion/clip-vit')

"""
HF_ENDPOINT=https://hf-mirror.com python download_clip.py
"""