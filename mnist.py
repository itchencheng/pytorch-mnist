#coding:utf-8

import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append('data/')
import MnistDataset
sys.path.append('model/')
import VGG
import ResNet

from torchvision import transforms, utils
from tensorboardX import SummaryWriter

import pandas as pd

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
#ckpt_name = "checkpoints/VGG16_MNIST.pth"
#log_name = "./logs/VGG16_MNIST_log/"
ckpt_name = "checkpoints/ResNet18_MNIST.pth"
log_name = "./logs/ResNet18_MNIST_log/"




def train(cnn_model, start_epoch, train_loader, test_loader):


    # train model from scratch
    num_epochs = 10
    learning_rate = 0.003

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=learning_rate)

    train_writer = SummaryWriter(log_dir=log_name+'train')
    test_writer = SummaryWriter(log_dir=log_name+'test')

    train_offset = 0
    for epc in range(num_epochs):
        
        epoch = epc + start_epoch

        for batch_idx, (data_x, data_y) in enumerate(train_loader):

            train_iter = train_offset + epoch * len(train_loader) + batch_idx

            data_x = data_x.to(device)
            data_y = data_y.to(device)

            optimizer.zero_grad()

            output = cnn_model(data_x)

            loss = criterion(output, data_y)
            loss.backward()
            optimizer.step()

            if (train_iter % 10 == 0):
                print("Epoch %d/%d, Step %d/%d, iter %d Loss: %f" %(epoch, start_epoch+num_epochs, batch_idx, len(train_loader), train_iter, loss.item()))
                train_writer.add_scalar('data/loss', loss, train_iter)


            if (train_iter % 100 == 0):

                with torch.no_grad():
                    correct = 0
                    total = 0

                    loss = 0
                    for test_batch_idx, (images, labels) in enumerate(test_loader):
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = cnn_model(images)
                        loss += criterion(outputs.squeeze(), labels.squeeze())
                        
                        _, predicted = torch.max(outputs.data, 1)

                        total += batch_size
                        correct += (predicted == labels).sum().item()
                    
                    loss = float(loss) / len(test_loader)
                    test_writer.add_scalar('data/loss', loss, train_iter)

                    acc = float(correct)/total

                    print("iter %d, Test Accuracy: %f" %(train_iter, acc))
                    print("iter %d, Test avg Loss: %f" %(train_iter, loss))

                    test_writer.add_scalar('data/accuracy', acc, train_iter)

        # save models
        state_dict = {"state": cnn_model.state_dict(), "epoch": epoch, "acc": acc}
        torch.save(state_dict, ckpt_name)
        print("Model saved! %s" %(ckpt_name))



def test(cnn_model, real_test_loader):
    labels = []

    for batch_idx, images in enumerate(real_test_loader):
        images = images.to(device)

        outputs = cnn_model(images)

        prob = torch.nn.functional.softmax(outputs.data)
        prob = prob.data.tolist()
        _, predicted = torch.max(outputs.data, 1)

        predicted = predicted.data.tolist()
        for item in predicted:
            labels.append(item)

    ids = range(1, len(labels)+1)

    submission = pd.DataFrame({'ImageId': ids, 'Label': labels})
    output_file_name = "submission.csv"
    submission.to_csv(output_file_name, index=False)
    print("# %s generated!" %(output_file_name))



def main():
    if (len(sys.argv) < 2):
        print("Error: usage: python main.py train/test!")
        exit(0)
    else:
        mode = sys.argv[1]

    print(mode)

    # enhance
    transform_enhanc_func = transforms.Compose([
        transforms.Pad(2),
        transforms.RandomRotation(5),
        transforms.ToTensor()
        ])
    # transform
    transform_func = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor()
        ])

    # model create
    # model = VGG.VGG("VGG16", 1).to(device)
    model = ResNet.ResNet18(1).to(device)
    print("Model created!")
    start_epoch = 0

    # model resume
    if (os.path.exists(ckpt_name)):
        status_dict = torch.load(ckpt_name)
        model_state = status_dict["state"]
        start_epoch = status_dict["epoch"] + 1
        acc = status_dict["acc"]
        model.load_state_dict(model_state)
        print("Model loaded!")

    # train
    if (mode == 'train'):
        train_data_path = '/home/chen/dataset/kaggle/digit-recognizer/train'
        train_dataset = MnistDataset.MnistDataset(train_data_path, True, False, transform_enhanc_func)
        val_dataset = MnistDataset.MnistDataset(train_data_path, True, True, transform_func)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        train(model, start_epoch, train_dataloader, val_dataloader)
    
    else:
        test_data_path = '/home/chen/dataset/kaggle/digit-recognizer/test'
        test_dataset = MnistDataset.MnistDataset(test_data_path, False, False, transform_func)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        print(len(test_dataset))

        test(model, test_dataloader)



if __name__ == "__main__":
    main()