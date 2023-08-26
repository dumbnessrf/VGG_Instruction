

import numpy

import torch
from torch.utils.data import DataLoader

import utils
from dataset import MyDataset

if __name__ == '__main__':
    onnx_session = utils.load_onnx_model("./checkpoints/mnist.onnx")
    input_name = onnx_session.get_inputs()[0].name
    input_shape = onnx_session.get_inputs()[0].shape
    test_list,_=utils.get_files(r"test_images",1)
    test_loader = DataLoader(MyDataset(test_list, transform=None, test=True), batch_size=1, shuffle=True,
                             collate_fn=utils.collate_fn)
    pros=[]
    ok_num=0
    for item in test_loader:
        image,label=item
        np_sample = numpy.array(image)

        result = onnx_session.run(None, {input_name: np_sample})

        softmax = torch.nn.Softmax(dim=2)
        arr=softmax(torch.from_numpy(numpy.array(result)))
        prop =torch.max(arr)
        pros.append(prop.item())
        res = torch.eq(torch.from_numpy(numpy.array(label)).long(), torch.argmax(torch.from_numpy(numpy.array(result))))
        if (res): ok_num = ok_num + 1
    print("[{}/{}],correct rate:{}".format(ok_num, len(test_list), ok_num / len(test_list)))