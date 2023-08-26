# 这个数据集类的作用就是加载训练和测试时的数据
import json

import cv2
from PIL.Image import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from utils import get_files, collate_fn, visualize_batch_tensor

with open('config.json') as f:
    param_dict = json.load(f)


class MyDataset(Dataset):
    def __init__(self, data, transform=None, test=False):
        """

        :param data:
        :param transform: Default ToTensor
        :param test: Test Dataset Ratio
        """
        imgs = []
        labels = []
        self.test = test
        self.len = len(data)
        self.data = data
        self.transform = transform
        for i in self.data:
            imgs.append(i[0])
            self.imgs = imgs
            labels.append(int(i[1]))  # pytorch中交叉熵需要从0开始
            self.labels = labels

    def __getitem__(self, index):
        if self.test:
            label = self.labels[index]
            img_path = self.imgs[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (param_dict["image_width"], param_dict["image_height"]))
            img = transforms.ToTensor()(img)
            return img, label
        else:
            img_path = self.imgs[index]
            label = self.labels[index]

            img = cv2.imread(img_path)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (param_dict["image_width"], param_dict["image_height"]))

            # 看有没有数据增强
            if self.transform is not None:
                img = Image.fromarray(img)
                img = self.transform(img)

            else:
                img = transforms.ToTensor()(img)

            return img, label

    def __len__(self):
        return len(self.data)  # self.len

if __name__ == '__main__':
    test_data, _ = get_files(param_dict["dataset_folder"], param_dict["test_data_ratio"])
    transform = transforms.Compose([transforms.ToTensor()])
    data = MyDataset(test_data, transform=transform)
    dataset_folder = param_dict["dataset_folder"]
    test_data_ratio = param_dict["test_data_ratio"]
    batch_size = param_dict["batch_size"]
    test_list, train_list = get_files(dataset_folder, test_data_ratio)
    train_loader = DataLoader(MyDataset(train_list, transform=None, test=False), batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn)  # 测试时这里的batch_size = 1
    test_loader = DataLoader(MyDataset(test_list, transform=None, test=True), batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)  # 测试时这里的batch_size = 1
    for index, (input, target) in enumerate(train_loader):
        print(input.shape)
        # 多张网格显示
        visualize_batch_tensor(input)
        # 单张显示

        pass
