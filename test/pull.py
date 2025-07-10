import torch
import torchvision
import os
from os.path import join as j_
from PIL import Image
import pandas as pd
import numpy as np

# loading all packages here to start
from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.eval_patch_features.linear_probe import eval_linear_probe
from uni.downstream.eval_patch_features.fewshot import eval_knn, eval_fewshot
from uni.downstream.eval_patch_features.protonet import ProtoNet, prototype_topk_vote
from uni.downstream.eval_patch_features.metrics import get_eval_metrics, print_metrics
from uni.downstream.utils import concat_images

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_model():
    # 下载模型的正确脚本 - 开始

    login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    # pretrained=True needed to load UNI2-h weights (and download weights for the first time)
    timm_kwargs = {
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval()
    # 下载模型的正确脚本 - 结束


if __name__ == '__main__':
    download_model()
    # persons = ['person1', 'person2', 'person3']
    # msg = ''
    # for person in persons:
    #     # 加载模型
    #     msg += person
    #
    # print(msg)
    # print(' '.join(persons))
