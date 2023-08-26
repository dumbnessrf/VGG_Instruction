from torch import  from_numpy,randn
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.utils.data import DataLoader
from numpy import  array
from json import load

from VGG import VGG
from dataset import MyDataset

from utils import get_files, collate_fn, evaluate, save_checkpoint,Convert_ONNX
from datetime import datetime


if __name__ == '__main__':


    with open('config.json') as f:
        param_dict = load(f)



    epochs = param_dict["epochs"]
    lr = param_dict["learning_rate"]
    lr_decay = param_dict["lr_decay"]
    weight_decay = param_dict["weight_decay"]
    batch_size = param_dict["batch_size"]
    dataset_folder = param_dict["dataset_folder"]
    test_data_ratio = param_dict["test_data_ratio"]
    num_classes = len(param_dict["class_labels"])
    image_channels = param_dict["image_channels"]
    Momentum = param_dict["Momentum"]
    weights = param_dict["weights"]
    model_name = param_dict["model_name"]
    val_step_interval = param_dict["val_step_interval"]
    train_step_interval = param_dict["train_step_interval"]
    model = VGG(image_channels,num_classes)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=Momentum)

    criterion = CrossEntropyLoss()

    test_list, train_list = get_files(dataset_folder, test_data_ratio)

    train_loader = DataLoader(MyDataset(train_list, transform=None, test=False), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    test_loader = DataLoader(MyDataset(test_list, transform=None, test=True), batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn)
    print("训练集数量{}", train_list.__len__())
    print("测试集数量{}", test_list.__len__())
    accuracies = []
    test_loss = []
    train_loss = []
    current_accuracy = 0
    model.train()
    for epoch in range(epochs):
        start_time = datetime.now()



        loss_epoch = 0

        for index, (input, target) in enumerate(train_loader):
            model.train()

            # else:
            input = (input)
            target = (from_numpy(array(target)).long())

            output = model(input)



            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()



        end_time = datetime.now()
        print("epoch:{},耗时: {}秒".format(epoch,end_time - start_time))
        if (epoch + 1) % train_step_interval == 0:
            print("Epoch: {} \t Loss: {:.6f} ".format(epoch + 1, loss_epoch))
        if (epoch + 1) % val_step_interval == 0:
            print("\n------ Evaluate ------")
            model.eval()


            test_loss1, accTop1 = evaluate(test_loader, model, criterion)
            accuracies.append(accTop1)
            test_loss.append(test_loss1)
            train_loss.append(loss_epoch / len(train_loader))
            print("Test_epoch: {} Test_accuracy: {:.4}% Test_Loss: {:.6f}".format(epoch + 1, accTop1, test_loss1))
            is_save_model = accTop1 > current_accuracy  # 测试的准确率大于当前准确率为True
            accTop1 = max(current_accuracy, accTop1)
            current_accuracy = accTop1
            save_checkpoint({
                "epoch": epoch + 1,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "accTop1": current_accuracy,
                "optimizer": optimizer.state_dict(),
            }, is_save_model, weights + model_name + ".pth", model_name)


    x=randn(1,param_dict["image_channels"],param_dict["image_height"],param_dict["image_width"])
    Convert_ONNX(model,x, "checkpoints/mnist.onnx")


