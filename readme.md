# MoonCast: High-Quality Zero-Shot Podcast Generation

## Overview
Demo page: [demo](https://mooncastdemo.github.io)

Paper: [paper](https://arxiv.org/abs/2503.14345)

We open-source this system to advance the field of human-like speech synthesis. Our goal is to create more natural and expressive synthetic voices that bridge the gap between machines and humans. We hope this project will inspire researchers and developers to explore new possibilities in voice technology. We warmly welcome contributions from anyone interested in this project. Whether through code, documentation, feedback, or sharing your insights, every input helps make this project better.

### Abstract
Recent advances in text-to-speech synthesis have achieved notable success in generating high-quality short utterances for individual speakers. However, these systems still face challenges when extending their capabilities to long, multi-speaker, and spontaneous dialogues, typical of real-world scenarios such as podcasts. These limitations arise from two primary challenges: 1) long speech: podcasts typically span several minutes, exceeding the upper limit of most existing work; 2) spontaneity: podcasts are marked by their spontaneous, oral nature, which sharply contrasts with formal, written contexts; existing works often fall short in capturing this spontaneity. In this paper, we propose MoonCast, a solution for high-quality zero-shot podcast generation, aiming to synthesize natural podcast-style speech from text-only sources (e.g., stories, technical reports, news in TXT, PDF, or Web URL formats) using the voices of unseen speakers. To generate long audio, we adopt a long-context language model-based audio modeling approach utilizing large-scale long-context speech data. To enhance spontaneity, we utilize a podcast generation module to generate scripts with spontaneous details, which have been empirically shown to be as crucial as the text-to-speech modeling itself. Experiments demonstrate that MoonCast outperforms baselines, with particularly notable improvements in spontaneity and coherence.

## Environment Setup
- Create conda environment.

``` sh
conda create -n mooncast -y python=3.10
conda activate mooncast
pip install -r requirements.txt 
pip install flash-attn --no-build-isolation
pip install huggingface_hub
```

- Download the pretrained weights.
``` sh
python download_pretrain.py
```

## Example Usage

The audio prompts used in this project are sourced from publicly available podcast segments and are intended solely for demonstration purposes. Redistribution of these audio files, whether in their original form or as generated audio, is strictly prohibited. If you have any concerns or questions regarding the use of these audio files, please contact us at juzeqian@mail.ustc.edu.cn

```sh
CUDA_VISIBLE_DEVICIES=0 python inference.py
```

## Disclaimer  
This project is intended for **research purposes only**. We strongly encourage users to **use this project and its generated audio responsibly**. **We are not responsible for any misuse or abuse of this project**. By using this project, you agree to comply with all applicable laws and ethical guidelines.