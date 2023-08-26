from math import ceil
from datetime import datetime


from onnxruntime import InferenceSession
from torchvision.utils import save_image,make_grid
from torch import  from_numpy,no_grad,stack,save,load,onnx
from numpy import  array,transpose,hstack,random

from matplotlib.pyplot import show,imshow
from os import listdir,scandir,mkdir,path

from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph


def visualize_batch_tensor(t1):
    # 传入的类型是torch.float32，不需要.long()
    # img=torchvision.utils.make_grid(t1.long()).numpy()
    img = make_grid(t1).numpy()

    imshow(transpose(img, (1, 2, 0)))
    show()


def lr_step(epoch):
    if epoch < 30:
        lr = 0.01
    elif epoch < 80:
        lr = 0.001
    elif epoch < 120:
        lr = 0.0005
    else:
        lr = 0.0001
    return lr


def evaluate(test_loader, model, criterion):
    """

    :param test_loader:test dataloader be tested
    :param model:model be tested
    :param criterion: loss function be used to calculate loss
    :return:return average loss,average accuracy
    """
    with no_grad():
        sum = 0
        test_loss_sum = 0
        test_top1_sum = 0
        model.eval()

        for item in test_loader:
            images, labels = item

            # if cuda.is_available():
            #     input_test = (images).cuda()
            #     target_test = (from_numpy(array(labels)).long()).cuda()
            # else:
            input_test = images
            target_test = (from_numpy(array(labels)).long())

            output_test = model(input_test)
            loss = criterion(output_test, target_test)
            # print(target_test)
            # print(output_test)
            top1_test = accuracy(output_test, target_test, topk_index=(1,))
            sum += 1
            test_loss_sum += loss.data.cpu().numpy()
            test_top1_sum += top1_test[0].cpu().numpy()[0]

        # 平均loss
        avg_loss = test_loss_sum / sum
        #
        avg_top1 = test_top1_sum / sum
        return avg_loss, avg_top1


def accuracy(output, target, topk_index=(1, 5)):
    '''
    计算模型的precision，范围为topk_index
    :param output:
    :param target:
    :param topk_index:
    :return:return this batch accuracy,correct num / total num
    '''
    maxk = max(topk_index)
    batch_size = target.size(0)  # size(0) = batch_size  size(1) = num_classes
    # 取output维度为1的maxk个最大值，若取1个最大值，维度为1，那么对于output的size来说[batch,num_classes]来说的话，维度为1就是在num_classes上取最大值，[[1,0,2],[3,0,1],[0,2,1]]=>[2,0,1]
    _, pred = output.topk(maxk, 1, True, True)  # 1 是dim维度
    # pred转为tensor
    pred = pred.t()  # 转置
    # target用view平铺再转为pred同样格式的tensor；然后pred与转换后的target做eq；返回由true，false组成的数组
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # eq表示是否相等

    res = []
    for k in topk_index:
        # calculate true labels count，it is need to view(-1) due to "correct" is tensor,
        # and convert all true or false labels to float which is needed to calculate count through sum()
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)  # correct[:k]是取前k行
        '''.float()转换成float类型，False = 0,True = 1'''
        # correct_k*(100/batch_size) can manifest the accuracy for this batch
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def collate_fn(batch):
    """
    表示如何将多个样本拼接成一个batch
    :param batch: 批次数量
    :return: [b,c,h,w]
    """
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return stack(imgs, 0), label


def save_model(save_path, model):
    save({'state_dict': model.state_dict()},
               save_path)


def load_model(save_name, model):
    model_data = load(save_name)
    model.load_state_dict(model_data['state_dict'])


def get_allfiles(file_dir):
    labels = [f.name for f in scandir(file_dir) if f.is_dir()]
    labels_folder = [f.path for f in scandir(file_dir) if f.is_dir()]
    images_dict = dict()
    labels_dict = dict()
    for i, (folder) in enumerate(labels_folder):
        images_dict[labels[i]] = list()
        labels_dict[labels[i]] = list()
        for file in listdir(folder):
            images_dict[labels[i]].append(folder + "\\" + file)
            labels_dict[labels[i]].append(labels[i])
    image_list = hstack([f for f in images_dict.values()])
    labels_list = hstack([f for f in labels_dict.values()])
    # np.array=>{input1,input2}
    temp = array([image_list, labels_list])
    temp = temp.transpose()
    random.shuffle(temp)
    return temp


def get_files(file_dir, ratio):
    """
    分别获取测试集和训练集图片
    :param file_dir:包含所有label的图像文件夹Dataset，Dataset=>{"apple","peach","lemon"}
    :param ratio:测试集比例
    :return: 返回{图片路径,label}
    """
    labels = [f.name for f in scandir(file_dir) if f.is_dir()]
    labels_folder = [f.path for f in scandir(file_dir) if f.is_dir()]
    images_dict = dict()
    labels_dict = dict()

    for i, (folder) in enumerate(labels_folder):
        images_dict[labels[i]] = list()
        labels_dict[labels[i]] = list()
        for file in listdir(folder):
            images_dict[labels[i]].append(folder + "\\" + file)
            labels_dict[labels[i]].append(labels[i])

    # for file  in os.listdir(file_dir +'roses'):
    #     roses.append(file_dir + 'roses' + '/' + file)
    #     labels_roses.append(0)
    # for file in os.listdir(file_dir + 'tulips'):
    #     tulips.append(file_dir + 'tulips' + '/' + file)
    #     labels_tulips.append(1)
    # for file in os.listdir(file_dir + 'dandelion'):
    #     tulips.append(file_dir + 'dandelion' + '/' +file)
    #     labels_dandelion.append(2)
    # for file in os.listdir(file_dir + 'sunflowers'):
    #     sunflowers.append(file_dir + 'sunflowers' + '/' +file)
    #     labels_sunflowers.append(3)

    image_list = hstack([f for f in images_dict.values()])
    labels_list = hstack([f for f in labels_dict.values()])
    # np.array=>{input1,input2}
    temp = array([image_list, labels_list])
    temp = temp.transpose()
    random.shuffle(temp)
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    all_label_list = [int(i) for i in all_label_list]
    length = len(all_image_list)
    n_test = int(ceil(length * ratio))
    n_train = length - n_test

    tra_image = all_image_list[0:n_train]
    tra_label = all_label_list[0:n_train]

    test_image = all_image_list[n_train:-1]
    test_label = all_label_list[n_train:-1]

    train_data = [(tra_image[i], tra_label[i]) for i in range(len(tra_image))]
    test_data = [(test_image[i], test_label[i]) for i in range(len(test_image))]
    # print("train_data = ",test_image)
    # print("test_data = " , test_label)
    return test_data, train_data


def save_checkpoint(state, is_save_model, filename, modelname):
    """

    :param state:example:\n
    save_checkpoint({
                "epoch": epoch + 1,
                "model_name": config.model_name,
                "state_dict": model.state_dict(),
                "accTop1": current_accuracy,
                "optimizer": optimizer.state_dict(),
            }, save_model)
    :param is_save_model:model to be saved
    :param filename:file to be written
    :param modelname:model name,be logged
    :return:
    """

    saveDir = path.dirname(filename)
    if not (path.exists(saveDir)):
        mkdir(saveDir)
    # save_model(state,)
    save(state, filename)
    if is_save_model:
        message = filename
        print("Get Better top1 : %s saving weights to %s" % (state["accTop1"], message))
        with open("./logs/%s.txt" % modelname, "a") as f:
            print("Get Better top1 : %s saving weights to %s" % (state["accTop1"], message), file=f)


# 转为ONNX
def Convert_ONNX(model,x, model_savepath=""):
    """
    convert model to onnx format and save to local diskd
    :param model: a net model to be saved
    :param x: simulate size
    :param model_savepath: file name to be written
    :return:
    """
    # 设置模型为推理模式
    model.eval()

    # 设置模型输入的尺寸
    # dummy_input = torch.randn(1, input_size, requires_grad=True)

    # 导出ONNX模型
    onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      model_savepath,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      input_names=['input'],  # the model's input names
                      output_names=['output']  # the model's output names
                      )
    print('Model has been converted to ONNX,path:{}'.format(model_savepath))


def ConvertDataLoader2LocalImage(dataloader, savefolder):
    """
    save images from dataloader to local image file
    :param dataloader: dataloader对象
    :param savefolder: a file folder to be saved;the folder will be created for every single label,and the image format determined by datetime now
    :return:
    """
    if not path.exists(savefolder):
        mkdir(savefolder)
    for item in dataloader:
        data, target = item

        save_folder = savefolder + "/" + str(target.item())
        if not path.exists(save_folder):
            mkdir(save_folder)
        save_path = save_folder + "/" + str(datetime.now().strftime('%Y_%m_%d %H_%M_%S_%f')) + ".png"
        save_image(data, save_path)

    # train_folder="train_images"
    # if not os.path.exists(train_folder):
    #     os.mkdir(train_folder)
    # for item in train_dataloader:
    #     data, target = item
    #
    #     save_folder = train_folder + "/" + str(target.item())
    #     if not os.path.exists(save_folder):
    #         os.mkdir(save_folder)
    #     save_path=save_folder+"/"+str(datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S_%f'))+".png"
    #     torchvision.utils.save_image(data,save_path)

def load_onnx_model(filename):

    try:
        session = InferenceSession(filename)

    except (InvalidGraph, TypeError, RuntimeError) as e:
        # It is possible for there to be a mismatch between the onnxruntime and the
        # version of the onnx model format.
        print(e)
        raise e

    return session

