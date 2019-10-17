import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time

from torch.autograd import Variable


def load_train():
    img_data = torchvision.datasets.ImageFolder(root = r'./train2' ,
                                                transform = transforms.Compose([
                                                transforms.Resize( 224 ),
                                                transforms.CenterCrop( 224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #R,G,B每層的歸一化
                                               )
    train_loader = torch.utils.data.DataLoader(dataset = img_data, batch_size=32,shuffle= True, num_workers=2)
    return train_loader

def load_test():
    path = r'./test3'
    testset = torchvision.datasets.ImageFolder(path ,
                                               transform = transforms.Compose([
                                               transforms.Resize( 224 ),
                                               transforms.CenterCrop( 224),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]) #R,G,B每層的歸一化
                                               )
    testloader = torch.utils.data.DataLoader(testset , batch_size = 32 ,shuffle = False , num_workers = 2 )
    return testloader

def reload_net():
    trainednet = models.resnet34(pretrained=True)  # pretrained表示是否加载已经与训练好的参数
    trainednet.fc = torch.nn.Linear(512, 13)  # 将最后的fc层的输出改为标签数量
    trainednet.load_state_dict(torch.load('./V3_resnet34_net_params.pth'))
    return trainednet


batch_size = 32
learning_rate = 0.00001
epoch = 300
classes = ('bedroom', 'coast', 'forest', 'highway', 'insidecity', 'kitchen', 'livingroom', 'mountain', 'office', 'opencountry', 'street', 'suburb', 'tallbuilding')

trainLoader = load_train()
testLoader = load_test()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#最一開始需要設定
#model = models.resnet34(pretrained=True)
#model.fc = torch.nn.Linear(512, 13) #將作後的fc層改為13層

model = reload_net() #有訓練過的model使用
model = model.to(device) #如果有GPU，而且確認使用則保留；如果没有GPU，使用CPU
criterion = torch.nn.CrossEntropyLoss() #定義損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #定優化函數


train_loss = []
valid_loss = []
accuracy = []

def evaluate():
    model.eval()
    corrects = eval_loss = 0

    for image, label in testLoader:
        image = Variable(image.to(device))
        label = Variable(label.to(device))
        pred = model(image)
        loss = criterion(pred, label)

        eval_loss += loss.item()
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
    return eval_loss/float(len(testLoader)), corrects, corrects*100.0/len(testLoader), len(testLoader)

def train():
    model.train()
    total_loss = 0
    for image, label in trainLoader:
        image = Variable(image.to(device))
        label = Variable(label.to(device))
        optimizer.zero_grad()

        target = model(image)
        loss = criterion(target, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss/float(len(trainLoader))

best_acc = None
total_start_time = time.time()

#第一次啟動程式 start 設為0
start = 1
fp = open('loss9.txt','r')
min = float(fp.read())
fp.close()

try:
    print('-' * 90)
    for epoch in range(1, epoch+1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss*1000.)
        if start == 0:
            start = 1
            min = loss

        if loss < min:
            min = loss
            torch.save(model, './V3_resnet34_net.pth')
            torch.save(model.state_dict(), './V3_resnet34_net_params.pth')
            fp = open('loss9.txt','w')
            fp.write(str(loss))
            fp.close()
            print("-" * 90)
            print("-----------Save---------------")

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(epoch,
                                                                    time.time() - epoch_start_time,
                                                                    loss))
        print('-' * 10)

except KeyboardInterrupt:
    print("-"*90)
    print("Exiting from training early | cost time: {:5.2f}min".format((time.time() - total_start_time)/60.0))

