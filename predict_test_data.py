import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd

from torch.autograd import Variable


def load_train():
    img_data = torchvision.datasets.ImageFolder(root = r'./train2' ,
                                                transform = transforms.Compose([
                                                transforms.Resize(224),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #R,G,B每層的歸一化
                                               )
    train_loader = torch.utils.data.DataLoader(dataset = img_data, batch_size=32,shuffle= True, num_workers=2)
    return train_loader

def load_test():
    path = r'./test2'
    testset = torchvision.datasets.ImageFolder(path ,
                                               transform = transforms.Compose([
                                               transforms.Resize( 224 ),
                                               transforms.CenterCrop( 224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #R,G,B每層的歸一化
                                              )
    testloader = torch.utils.data.DataLoader(testset , batch_size = 32 ,shuffle = False , num_workers = 2 )
    return testloader

def reload_net():
    trainednet = models.resnet34(pretrained=True)  # 使用pytorch resnet34模型
    trainednet.fc = torch.nn.Linear(512, 13)  # fc層改為標籤數量
    trainednet.load_state_dict(torch.load('./V3_resnet34_net_params.pth'))
    return trainednet


batch_size = 32
learning_rate = 0.01
epoch = 100


classes = ('bedroom', 'coast', 'forest', 'highway', 'insidecity', 'kitchen', 'livingroom', 'mountain', 'office', 'opencountry', 'street', 'suburb', 'tallbuilding')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = reload_net()
model = model.to(device) #使用GPU 或CPU

testloader = load_test()
dataiter = iter (testloader)
images , labels = dataiter.next ()
images , labels = images.to(device), labels.to(device)
outputs = model(Variable(images))
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ' , " ".join('%5s' % classes[predicted[j]] for j in range (len(predicted))))# 打印前25個預測值

answer = []
num = 0
while True:
    try:
        for j in range(len(predicted)):
            print(num)
            num = num + 1
            answer.append(classes[predicted[j]])
        images , labels = dataiter.next()
        images , labels = images.to(device), labels.to(device)
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
    except:
            break

out = pd.read_csv(r'./sameple_submission.csv')
ans = pd.Series(answer)
out.label = ans
out.to_csv('./out_3.csv', index = False, sep = ',')