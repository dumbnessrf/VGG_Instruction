import cv2
import numpy
import torch.nn
from torch import softmax
from utils import get_files

if __name__ == '__main__':
    net = cv2.dnn.readNetFromONNX("checkpoints/mnist.onnx")  # 加载训练好的识别模型
    test_list, train_list = get_files(r"test_images", 1)

    pros=[]
    ok_num=0
    predicts=[]
    for item in test_list:
        image_path,label=item
        # image = cv2.imread(image_path,flags=cv2.IMREAD_GRAYSCALE)  # 读取图片
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.imread(image_path)
        image=cv2.resize(image,  dsize=(32, 32))
        blob = cv2.dnn.blobFromImage(image)  # 由图片加载数据 这里还可以进行缩放、归一化等预处理
        net.setInput(blob)  # 设置模型输入
        out = net.forward()  # 推理出结果
        pro=max(softmax(torch.from_numpy(out),1,dtype=torch.float))
        pros.append( pro)
        predict=torch.argmax( softmax(torch.from_numpy(out),1,dtype=torch.float)).item()
        res = torch.eq(torch.from_numpy(numpy.array(label)).long(), torch.argmax(torch.from_numpy(out)))
        if(res):ok_num=ok_num+1
        predicts.append( predict)

    print("[{}/{}],correct rate:{}".format(ok_num,len(test_list),ok_num/len(test_list)))
