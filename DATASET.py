
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os


class CMDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        partition='train',
        data_name='data_name'
    ):
        self.data_name = data_name
        self.partition = partition
        self.open_data()

    def open_data(self):
        if self.data_name.lower() == 'mirflickr25k_deep':
            self.imgs, self.texts, self.labels, self.label_vec = MIRFlickr25K_fea(self.partition)
        elif self.data_name.lower() == 'iaprt_tc12':
            self.imgs, self.texts, self.labels = IAPR_fea(self.partition)
        elif self.data_name.lower() == 'nus_wide_deep':
            self.imgs, self.texts, self.labels, self.label_vec = NUSWIDE_fea(self.partition)
        elif self.data_name.lower() == 'mscoco_deep':
            self.imgs, self.texts, self.labels, self.label_vec = MSCOCO_fea(self.partition)
        self.length = self.labels.shape[0]
        self.text_dim = self.texts.shape[1]
        self.label_dim = self.labels.shape[1]

    def __getitem__(self, index):
        image = self.imgs[index]
        text = self.texts[index]
        label = self.labels[index]
        return image, text, label

    def __len__(self):
        return self.length


def MIRFlickr25K_fea(partition):
    # root = '/media/hdd4/liy/data/MIRFLICKR25K'
    root = './data/MIRFLICKR25K/MIRFLICKR25K'
    data_img = sio.loadmat(os.path.join(
        # root, 'mirflickr25k-iall-vgg-rand.mat'))['XAll']
        root, 'mirflickr25k-iall-vgg.mat'))['XAll']
    data_txt = sio.loadmat(os.path.join(
        # root, 'mirflickr25k-yall-rand.mat'))['YAll']
        root, 'mirflickr25k-yall.mat'))['YAll']
    labels = sio.loadmat(os.path.join(
        # root, 'mirflickr25k-lall-rand.mat'))['LAll']
        root, 'mirflickr25k-lall.mat'))['LAll']
    label_vec = sio.loadmat(os.path.join(
        root, 'word_vec_flickr.mat'))['word_vec_flickr']
    test_size = 2000
    train_size = 10000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size:
            :], data_txt[-test_size::], labels[-test_size::]
    elif 'train' in partition.lower():
        data_img, data_txt, labels = data_img[0:
                                              train_size], data_txt[0: train_size], labels[0: train_size]
    else:
        data_img, data_txt, labels = data_img[0: -
                                              test_size], data_txt[0: -test_size], labels[0: -test_size]

    return data_img, data_txt, labels, label_vec


def IAPR_fea(partition):
    root = './data/IAPR-TC12/'
    file_path = os.path.join(root, 'iapr-tc12-rand.mat')
    data = sio.loadmat(file_path)

    valid_img = data['VDatabase'].astype('float32')
    valid_txt = data['YDatabase'].astype('float32')
    valid_labels = data['databaseL']

    test_img = data['VTest'].astype('float32')
    test_txt = data['YTest'].astype('float32')
    test_labels = data['testL']

    data_img, data_txt, labels = np.concatenate([valid_img, test_img]), np.concatenate(
        [valid_txt, test_txt]), np.concatenate([valid_labels, test_labels])

    test_size = 2000
    train_size = 10000
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size:
            :], data_txt[-test_size::], labels[-test_size::]
    elif 'train' in partition.lower():
        data_img, data_txt, labels = data_img[0:
                                              train_size], data_txt[0: train_size], labels[0: train_size]
    else:
        data_img, data_txt, labels = data_img[0: -
                                              test_size], data_txt[0: -test_size], labels[0: -test_size]
    return data_img, data_txt, labels


def NUSWIDE_fea(partition):
    # root = './data/NUS-WIDE-TC21/'
    root = './data/NUS-WIDE-TC21/NUS-WIDE-TC10/'
    # data_img = sio.loadmat(root + 'nus-wide-tc21-xall-vgg.mat')['XAll']
    data_img = sio.loadmat(root + 'nus-wide-tc10-xall-vgg.mat')['XAll']
    # data_txt = sio.loadmat(root + 'nus-wide-tc21-yall.mat')['YAll'][()]
    data_txt = sio.loadmat(root + 'nus-wide-tc10-yall.mat')['YAll'][()]
    # labels = sio.loadmat(root + 'nus-wide-tc21-lall.mat')['LAll']
    labels = sio.loadmat(root + 'nus-wide-tc10-lall.mat')['LAll']
    label_vec = sio.loadmat(os.path.join(
        root, 'word_vec_nuswide_tc10.mat'))['word_vec_nuswide_tc10']
    test_size = 2000
    train_size = 10500

    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size:
            :], data_txt[-test_size::], labels[-test_size::]
    elif 'train' in partition.lower():
        data_img, data_txt, labels = data_img[0:
                                              train_size], data_txt[0: train_size], labels[0: train_size]
    else:
        data_img, data_txt, labels = data_img[0: -
                                              test_size], data_txt[0: -test_size], labels[0: -test_size]
    return data_img, data_txt, labels, label_vec


# def MSCOCO_fea(partition):
#     root = './data/MSCOCO/'
#     import h5py
#
#     # path = root + 'MSCOCO_deep_doc2vec_data.h5py'
#     path = root + 'MSCOCO_deep_doc2vec_data_rand.h5py'
#     data = h5py.File(path)
#     data_img = np.concatenate(
#         [data['train_imgs_deep'][()], data['test_imgs_deep'][()]], axis=0)
#     data_txt = np.concatenate(
#         [data['train_text'][()], data['test_text'][()]], axis=0)
#     labels = np.concatenate(
#         [data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
#     test_size = 5000
#     train_size = 10000
#
#     if 'test' in partition.lower():
#         data_img, data_txt, labels = data_img[-test_size:
#             :], data_txt[-test_size::], labels[-test_size::]
#     elif 'train' in partition.lower():
#         data_img, data_txt, labels = data_img[0:
#                                               train_size], data_txt[0: train_size], labels[0: train_size]
#     else:
#         data_img, data_txt, labels = data_img[0: -
#                                               test_size], data_txt[0: -test_size], labels[0: -test_size]
#     return data_img, data_txt, labels


def MSCOCO_fea(partition):
    import h5py

    root = './data/MSCOCO/'
    path = root + 'MSCOCO_deep_doc2vec_data_rand.h5py'

    with h5py.File(path, 'r') as data:
        # 文件中只有 XAll, YAll, LAll
        data_img = data['XAll'][()]
        data_txt = data['YAll'][()]
        labels = data['LAll'][()]

    label_vec = sio.loadmat(os.path.join(
        root, 'word_vec_mscoco.mat'))['word_vec_mscoco']

    total_size = labels.shape[0]
    test_size = 5000
    train_size = 10000

    # 由于数据是随机排列的（文件名里有 rand），这里的切分方式假设前一部分为训练集，后一部分为测试集
    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img[-test_size:], data_txt[-test_size:], labels[-test_size:]
    elif 'train' in partition.lower():
        data_img, data_txt, labels = data_img[:train_size], data_txt[:train_size], labels[:train_size]
    else:  # retrieval 或全部数据
        data_img, data_txt, labels = data_img[:-test_size], data_txt[:-test_size], labels[:-test_size]

    return data_img, data_txt, labels, label_vec
