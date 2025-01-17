## Imports
import torch
import torch.nn.functional as F
import torch.nn as nn


input_size = 42*42
num_classes = 6

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

        self.conv0 = nn.Conv2d(1, 10, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(10, 20, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 40, 3, stride=1, padding=1)

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(42, 40)
        self.linear2 = nn.Linear(40, 30)
        self.linear3 = nn.Linear(30, 20)
        self.linear4 = nn.Linear(20, 10)
        self.linear5 = nn.Linear(10, num_classes)

        self.act = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2,2)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x, g):
        # print(xb.shape)
        out = self.conv0(x)
        out = self.act(out)
        # print(out.shape)
        out = self.conv1(out)
        out = self.act(out)
        # print(out.shape)
        out = self.conv2(out)
        out = self.act(out)
        # print(out.shape)

        out = self.adaptivepool(out)
        # print(out.shape)
        out = torch.cat((out, g), 1)
        # print(out.shape)

        out = self.flat(out)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.linear3(out)
        out = self.act(out)
        out = self.linear4(out)
        out = self.act(out)
        out = self.linear5(out)
        return(out)
    
    def nn_brain(self, obs):
        img_as_img = obs
        goal_np = img_as_img[:2]
        # goal_np = np.array([0.,0.])
        goal_np /= 1000.0
        img_as_img = img_as_img[2:]
        img_as_img /= 10000.0
        img_as_tensor = torch.from_numpy(img_as_img.astype('float32'))
        img_as_tensor = torch.unsqueeze(img_as_tensor, 0)
        img_as_tensor = torch.cat(tuple([img_as_tensor for item in range(len(img_as_img))]), 0)
        img_as_tensor = torch.unsqueeze(img_as_tensor, 0)
        img_as_tensor = torch.unsqueeze(img_as_tensor, 0)

        goal_as_tensor = torch.from_numpy(goal_np.astype('float32'))
        goal_as_tensor = torch.unsqueeze(goal_as_tensor, 1)
        goal_as_tensor = torch.unsqueeze(goal_as_tensor, 1)
        goal_as_tensor = torch.unsqueeze(goal_as_tensor, 0)

        # print(img_as_tensor.shape, goal_as_tensor.shape)
        output = self.model.forward(img_as_tensor, goal_as_tensor)
        # print(output)
        action = F.softmax(output).detach().numpy().argmax()
        return action

    def training_step(self, batch):
        images, goals, labels = batch
        out = self(images, goals) ## Generate predictions
        loss = self.loss_fn(out, labels) ## Calculate the loss
        return(loss)
    
    def validation_step(self, batch):
        images, goals, labels = batch
        out = self(images, goals)
        # labels = F.one_hot(labels, num_classes)
        # out = torch.argmax(out, dim=1) 
        # out = torch.argmax(out, dim=1) 
        # print(out.shape, labels.shape)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return({'val_loss':loss, 'val_acc': acc})
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})
    
    def epoch_end(self, epoch,result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))