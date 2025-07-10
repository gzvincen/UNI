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

from uni import get_encoder
import time


def do_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = get_encoder(enc_name='uni2-h', device=device)

    dataroot = '../assets/data/CRC100K/'
    assert os.path.isdir('../assets/data/CRC100K/NCT-CRC-HE-100K-NONORM')
    assert os.path.isdir('../assets/data/CRC100K/CRC-VAL-HE-7K')

    # get path to example data
    start = time.time()

    # create some image folder datasets for train/test and their data laoders
    train_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'NCT-CRC-HE-100K-NONORM'), transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'CRC-VAL-HE-7K'), transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=16)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16)

    # extract patch features from the train and test datasets (returns dictionary of embeddings and labels)
    train_features = extract_patch_features_from_dataloader(model, train_dataloader)
    test_features = extract_patch_features_from_dataloader(model, test_dataloader)

    # convert these to torch
    train_feats = torch.Tensor(train_features['embeddings'])
    train_labels = torch.Tensor(train_features['labels']).type(torch.long)
    test_feats = torch.Tensor(test_features['embeddings'])
    test_labels = torch.Tensor(test_features['labels']).type(torch.long)
    elapsed = time.time() - start
    print(f'Took {elapsed:.03f} seconds')

    linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
        train_feats=train_feats,
        train_labels=train_labels,
        valid_feats=None,
        valid_labels=None,
        test_feats=test_feats,
        test_labels=test_labels,
        max_iter=1000,
        verbose=True,
    )

    print_metrics(linprobe_eval_metrics)


def uni_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = get_encoder(enc_name='uni2-h', device=device)
    # get path to example data
    dataroot = '/Users/gaozhen/d_pan/Documents/source_code/UNI/assets/data/tcga_luadlusc'

    # create some image folder datasets for train/test and their data laoders
    train_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'train'), transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(j_(dataroot, 'test'), transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    # extract patch features from the train and test datasets (returns dictionary of embeddings and labels)
    train_features = extract_patch_features_from_dataloader(model, train_dataloader)
    test_features = extract_patch_features_from_dataloader(model, test_dataloader)

    # convert these to torch
    train_feats = torch.Tensor(train_features['embeddings'])
    train_labels = torch.Tensor(train_features['labels']).type(torch.long)
    test_feats = torch.Tensor(test_features['embeddings'])
    test_labels = torch.Tensor(test_features['labels']).type(torch.long)

    # fitting the model
    proto_clf = ProtoNet(metric='L2', center_feats=True, normalize_feats=True)
    proto_clf.fit(train_feats, train_labels)
    print('What our prototypes look like', proto_clf.prototype_embeddings.shape)

    # evaluating the model
    test_pred = proto_clf.predict(test_feats)
    get_eval_metrics(test_labels, test_pred, get_report=False)

    # 要报错，暂不清楚什么原因导致的：
    '''
    /opt/anaconda3/envs/UNI/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
        warnings.warn('resource_tracker: There appear to be %d '
    '''
    dist, topk_inds = proto_clf._get_topk_prototypes_inds(test_feats[:10], topk=2)


if __name__ == '__main__':
    # do_test()
    uni_test()
    print('Done')
