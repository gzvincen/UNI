import numpy as np
import torch
import torch.multiprocessing
from tqdm import tqdm

from ..get_encoder import get_encoder

torch.multiprocessing.set_sharing_strategy("file_system")

@torch.no_grad()
def extract_patch_features_from_dataloader(model, dataloader):
    """Uses model to extract features+labels from images iterated over the dataloader.

    Args:
        model (torch.nn): torch.nn CNN/VIT architecture with pretrained weights that extracts d-dim features.
        dataloader (torch.utils.data.DataLoader): torch.utils.data.DataLoader object of N images.

    Returns:
        dict: Dictionary object that contains (1) [N x D]-dim np.array of feature embeddings, and (2) [N x 1]-dim np.array of labels

    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters())[0].device

    for batch_idx, (batch, target) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu()[:remaining, :].cpu()
            labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
    }

    return asset_dict


import os
@torch.no_grad()
def extract_patch_features_from_dataloader_split(model, dataloader):
    """
        和上面方法差不多，只不过受限于 Mac 的内存，以及没有显卡的缘故，只能分批保存数据，然后再合并。
    """
    all_embeddings, all_labels = [], []
    batch_size = dataloader.batch_size
    device = next(model.parameters())[0].device

    output_dir = "/Users/gaozhen/d_pan/Documents/source_code/UNI/assets/data/CRC100K"
    count = 0

    for batch_idx, (batch, target) in tqdm(
            enumerate(dataloader), total=len(dataloader)
    ):
        remaining = batch.shape[0]
        if remaining != batch_size:
            _ = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, _])

        batch = batch.to(device)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu()[:remaining, :].cpu()
            labels = target.numpy()[:remaining]
            assert not torch.isnan(embeddings).any()

        # all_embeddings.append(embeddings)
        # all_labels.append(labels)
        torch.save(embeddings, os.path.join(output_dir, f'embeddings_batch_{batch_idx}.pt'))
        np.save(os.path.join(output_dir, f'labels_batch_{batch_idx}.npy'), labels)

        del embeddings
        del labels
        # if count == 0:
        #     print("直接跳出")
        #     break

    # asset_dict = {
    #     "embeddings": np.vstack(all_embeddings).astype(np.float32),
    #     "labels": np.concatenate(all_labels),
    # }

    return None

def many_to_one():
    # 自定义排序函数
    def extract_number(filename):
        # 提取文件名中的数字部分
        return int(filename.split('_')[-1].split('.')[0])

    output_dir = "/Users/gaozhen/d_pan/Documents/source_code/UNI/assets/data/CRC100K"
    embedding_files = sorted([f for f in os.listdir(output_dir) if f.startswith("embeddings_batch_")],
                             key=extract_number)
    label_files = sorted([f for f in os.listdir(output_dir) if f.startswith("labels_batch_")], key=extract_number)

    # [print(file) for file in embedding_files]
    # [print(file) for file in label_files]
    # return

    # all_embeddings = []
    all_labels = []
    handld_embeddings = None

    for index, embedding_file in enumerate(embedding_files):
        print(f"正在处理第{index}个batch, embedding_file: {embedding_file}")
        embeddings = torch.load(os.path.join(output_dir, embedding_file))
        # all_embeddings.append(embeddings)
        if handld_embeddings is None:
            handld_embeddings = np.vstack(embeddings).astype(np.float32)
        else:
            handld_ = np.vstack(embeddings).astype(np.float32)
            handld_embeddings = np.vstack([handld_embeddings, handld_])
        del embeddings
        # if index == 9:
        #     break

    # astype = np.vstack(all_embeddings).astype(np.float32)
    # print(astype.equals(handld_embeddings))
    # print("down")
    # return

    for index, label_file in enumerate(label_files):
        print(f"正在处理第{index}个batch, label_file: {label_file}")
        labels = np.load(os.path.join(output_dir, label_file))
        all_labels.append(labels)

    train_features = {
        # "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "embeddings": handld_embeddings,
        "labels": np.concatenate(all_labels),
    }

    torch.save(train_features, os.path.join(output_dir, 'train_features.pt'))


if __name__ == "__main__":
    many_to_one()
    print("Down")