import os

os.environ["LDFLAGS"] = "-L/usr/local/opt/libomp/lib"
os.environ["CPPFLAGS"] = "-I/usr/local/opt/libomp/include"
os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torchvision
# import os
from os.path import join as j_

from IPython.core.display_functions import display
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
import time
import matplotlib.pyplot as plt

# import os
# os.environ["OMP_NUM_THREADS"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    注意： Mac 运行 UNI 的解决方案
    1. Mac 安装 libomp
        brew install libomp
    
    2. 在终端设置环境变量，或者直接在代码中设置
        import os
        os.environ["LDFLAGS"] = "-L/usr/local/opt/libomp/lib"
        os.environ["CPPFLAGS"] = "-I/usr/local/opt/libomp/include"
        os.environ["DYLD_LIBRARY_PATH"] = "/usr/local/opt/libomp/lib:" + os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["OMP_NUM_THREADS"] = "1"
        
    3. 关注一下 /opt/anaconda3/envs/UNI310/lib 路径下的几个文件：
        $ ll | grep libomp
            lrwxr-xr-x   1 gaozhen staff   38  7 10 14:24 libgomp.1.dylib -> /usr/local/opt/libomp/lib/libomp.dylib
            lrwxr-xr-x   1 gaozhen staff   38  7 10 14:24 libgomp.dylib -> /usr/local/opt/libomp/lib/libomp.dylib
            lrwxr-xr-x   1 gaozhen staff   38  7 10 11:56 libiomp5.dylib -> /usr/local/opt/libomp/lib/libomp.dylib
            lrwxr-xr-x   1 gaozhen staff   38  7 10 14:27 libomp.dylib -> /usr/local/opt/libomp/lib/libomp.dylib
        这些文件最好都关联到 brew 安装的 libomp 路径下，否则会出现找不到 libomp 的问题。也可以先不管
        
    4. 卸载 pip 安装的 torch torchvision, 用 conda 安装 pytorch torchvision
        $ pip uninstall torch torchvision
        $ conda install -c pytorch pytorch torchvision
"""

def pt_train():
    # 下载模型的正确脚本 - 开始
    # torch_load = torch.load("/Users/gaozhen/d_pan/Documents/source_code/UNI/assets/data/CRC100K/embeddings_batch_0.pt")
    # load = np.load("/Users/gaozhen/d_pan/Documents/source_code/UNI/assets/data/CRC100K/labels_batch_0.npy")
    print('开始跑')
    model, transform = get_encoder(enc_name='uni2-h', device=device)
    # try:
    # except Exception as e:
    #     print('模型下载失败', e)
    assert os.path.isdir('../assets/data/CRC100K/NCT-CRC-HE-100K-NONORM')
    assert os.path.isdir('../assets/data/CRC100K/CRC-VAL-HE-7K')

    # get path to example data
    start = time.time()
    dataroot = '../assets/data/CRC100K/'

    # create some image folder datasets for train/test and their data laoders
    train_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'NCT-CRC-HE-100K-NONORM'), transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'CRC-VAL-HE-7K'), transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=16)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)

    # extract patch features from the train and test datasets (returns dictionary of embeddings and labels)
    # train_features = extract_patch_features_from_dataloader(model, train_dataloader)
    # torch.save(train_features, f'{dataroot}train_features.pt')
    train_features = torch.load(f"{dataroot}/train_features.pt", map_location=torch.device('cpu'))
    # test_features = extract_patch_features_from_dataloader(model, test_dataloader) 
    # torch.save(test_features, f'{dataroot}test_features.pt')
    test_features = torch.load(f"{dataroot}/test_features.pt", map_location=torch.device('cpu'))

    # convert these to torch
    train_feats = torch.Tensor(train_features['embeddings'])
    train_labels = torch.Tensor(train_features['labels']).type(torch.long)
    test_feats = torch.Tensor(test_features['embeddings'])
    test_labels = torch.Tensor(test_features['labels']).type(torch.long)
    elapsed = time.time() - start
    print(f'Took {elapsed:.03f} seconds')

    # fitting the model
    proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
    proto_clf.fit(train_feats, train_labels)
    print('What our prototypes look like', proto_clf.prototype_embeddings.shape)

    # evaluating the model
    test_pred = proto_clf.predict(test_feats)
    get_eval_metrics(test_labels, test_pred, get_report=False)

    dist, topk_inds = proto_clf._get_topk_queries_inds(test_feats, topk=5)
    print('label2idx correspondenes', test_dataset.class_to_idx)
    test_imgs_df = pd.DataFrame(test_dataset.imgs, columns=['path', 'label'])

    print('Top-k ADIPOSE-like test samples to ADIPOSE prototype')
    adi_topk_inds = topk_inds[0]
    adi_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][adi_topk_inds]],
                                  scale=0.5, gap=5)
    display(adi_topk_imgs)
    display_img(adi_topk_imgs)

    print('Top-k LYMPHOCYTE-like test samples to LYMPHOCYTE prototype')
    lym_topk_inds = topk_inds[3]
    lym_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][lym_topk_inds]],
                                  scale=0.5, gap=5)
    display(lym_topk_imgs)
    display_img(lym_topk_imgs)

    print('Top-k MUCOSA-like test samples to MUCOSA prototype')
    muc_topk_inds = topk_inds[4]
    muc_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][muc_topk_inds]],
                                  scale=0.5, gap=5)
    display(muc_topk_imgs)
    display_img(muc_topk_imgs)

    print('Top-k MUSCLE-like test samples to MUSCLE prototype')
    mus_topk_inds = topk_inds[5]
    mus_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][mus_topk_inds]],
                                  scale=0.5, gap=5)
    display(mus_topk_imgs)
    display_img(mus_topk_imgs)

    print('Top-k NORMAL-like test samples to NORMAL prototype')
    norm_topk_inds = topk_inds[6]
    norm_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][norm_topk_inds]],
                                   scale=0.5, gap=5)
    display(norm_topk_imgs)
    display_img(norm_topk_imgs)

    print('Top-k STROMA-like test samples to STROMA prototype')
    str_topk_inds = topk_inds[7]
    str_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][str_topk_inds]],
                                  scale=0.5, gap=5)
    display(str_topk_imgs)
    display_img(str_topk_imgs)

    print('Top-k TUMOR-like test samples to TUMOR prototype')
    tum_topk_inds = topk_inds[8]
    tum_topk_imgs = concat_images([Image.open(img_fpath) for img_fpath in test_imgs_df['path'][tum_topk_inds]],
                                  scale=0.5, gap=5)
    display(tum_topk_imgs)
    display_img(tum_topk_imgs)
    # 下载模型的正确脚本 - 结束


def display_img(imgs):
    plt.imshow(imgs)
    plt.axis('off')  # 可选，隐藏坐标轴
    plt.show()  # 这是关键，会弹出窗口显示图像


if __name__ == '__main__':
    pt_train()
