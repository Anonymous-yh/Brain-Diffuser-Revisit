import os
 
# 下载模型
os.system('huggingface-cli download --token *********************************** '
'--resume-download --local-dir-use-symlinks False shi-labs/versatile-diffusion '
'--local-dir /data01/yihan/brain-diffuser/versatile_diffusion/pretrained')