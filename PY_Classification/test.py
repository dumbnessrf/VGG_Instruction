import json
import numpy
import torch
from torch.utils.data import DataLoader
import utils
from VGG import VGG
from dataset import MyDataset

if __name__ == '__main__':
    with open('config.json') as f:
        param_dict = json.load(f)
    class_dict = dict()
    for i in range(len(param_dict["class_labels"])):
        class_dict[i] = param_dict["class_labels"][i]

    print("test class dict{}", class_dict)
    num_classes = len(param_dict["class_labels"])
    image_channels = param_dict["image_channels"]
    model = VGG(image_channels, num_classes)
    utils.load_model("checkpoints/mnist.pth", model)
    print(model)

    test_list = utils.get_allfiles(
        r"test_images")

    test_loader = DataLoader(MyDataset(test_list, transform=None, test=True), batch_size=1, shuffle=True,
                             collate_fn=utils.collate_fn)
    correct_num = 0
    step = 0
    total_num = len(test_list)
    with torch.no_grad():
        for item in test_loader:
            image, label = item

            output = model(image)
            # print(class_dict.__getitem__(numpy.argmax(output.numpy())))
            # label是list类型，需要转成tensor，output输出n分类的得分，需要求最大下标
            res = torch.eq(torch.from_numpy(numpy.array(label)).long(), torch.argmax(output))
            step = step + 1
            if (res):
                correct_num = correct_num + 1
            if (step % 100 == 0):
                print("{}/{},current accuracy{:.4f}".format(step, total_num, correct_num / step))
    print("[{}/{}],correct rate:{}".format(correct_num, len(test_list), correct_num / len(test_list)))
