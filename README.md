# 简介
本项目是一个基于 TensorFlowTTS 的中文语音合成 Demo
TensorFlowTTS是一个离线、开源的语音合成（text to speech)模型。它支持多种最前沿的模型选择，具备SOTA级效果。

# 环境配置与程序运行
## 基础环境
- Nvidia RTX2060 显卡
- windows 10 64位
- Anaconda3-2020.11-Windows-x86_64

## 配置 conda 环境并进行安装
### 配置conda清华源
conda config --set show_channel_urls yes
进入C:\Users\用户名，备份.condarc 文件后并用记事本方式打开
复制以下内容，覆盖源文件内容后保存
```
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```
### 创建名为 tf-tts 的 conda 环境
conda create -n tf-tts  python=3.8
conda activate tf-tts

### 下载源码(本代码中已经下载，可省略该步骤)
使用 git 下载 tensorflow-tts 代码
git clone https://github.com/TensorSpeech/TensorFlowTTS.git

### 安装依赖
进入到
pip install . -i https://pypi.douban.com/simple/
conda install cudatoolkit=10.1 cudnn=7.6.5 pyaudio

### nltk_data 配置
将 nltk_data.zip 解压后放到 C:\Users\用户名\AppData\Roaming 目录下

## 模型
本案例采用的是 tacotron2 和 mb.melgan，模型已经下载在工程目录，针对中文，可直接使用
由于 tacotron2 模型较大，所以进行了压缩，使用时需要先解压缩

## 运行程序
python tts-demo.py
正常会保存

