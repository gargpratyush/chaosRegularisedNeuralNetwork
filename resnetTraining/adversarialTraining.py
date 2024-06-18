# %% [markdown]
# # Adversarial Training on CIFAR-10 with FGSM attacks

# %% [markdown]
# In this notebook, we perform FGSM (targeted and non-targeted) attacks on the CIFAR-10 dataset using the Resnet18 model and build models to defend against these attacks using the Adversarial Training mechanism.

# %% [markdown]
# ## Table of contents
# *  **Preparing train and test data and building Resnet model**
# 
#     -  Training function
# 
#     -  Test function
# 
#     -  Trainig and test loss of Resnet18 model on Cifar-10
# 
# 
# *  **FGSM**
# 
#     - Visualizing 10 selected samples from the dataset
# 
#     - FGSM attack function
# 
#     -  Creating adversarial examples from samples with the FGSM attack and eps = 1/255
# 
#     -  Adversarial Training with FGSM
# 
#     - Comparing naturally-trained and adversarially-trained models
# 
#     - Evaluating the adversarially-trained model with FGSM against FGSM attack on test data
# 
#     - Targeted FGSM
# 

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import *

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from resnet import *

import pickle


# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.01

# %%
import torch

if torch.cuda.is_available():
    print("PyTorch is using GPU.")
    print("Number of GPUs available:", torch.cuda.device_count())
else:
    print("PyTorch is using CPU.")

# %% [markdown]
# <a name='name'></a>
# ### Preparing train and test data and building Resnet model

# %%
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# %% [markdown]
# ### Training function

# %%
def train(epoch, net):

    '''
    this function train net on training dataset
    '''

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss/len(trainloader)

# %% [markdown]
# ### Test function

# %%
def test(epoch, net):

    '''
    This function evaluate net on test dataset
    '''

    global acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100 * correct / total
    return test_loss/len(testloader)

# %%
train_losses=[]
test_losses=[]
epochs=15

for epoch in range(0,epochs):
    print(epoch)
    train_losses.append(train(epoch, net))
    test_losses.append(test(epoch, net))
    scheduler.step()
    

# %%
print('Accuracy of the network on the test images: %d %%' % (acc))

# %% [markdown]
# ####Training and test loss of Resnet18 model on Cifar-10

# %%
epochs=15
plt.plot(np.arange(1,epochs+1),train_losses, label='train losses')
plt.plot(np.arange(1,epochs+1), test_losses, label='test losses')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.legend()
plt.show()

# %% [markdown]
# ## FGSM

# %% [markdown]
# ### Visualizing 10 selected samples from the dataset

# %% [markdown]
# We need these samples later to make adversarial examples.

# %%
imgloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
dataiter = iter(imgloader)
org_images, org_labels = next(dataiter)

# %%
org_labels = org_labels.to(device)
org_images = org_images.to(device)
print(org_images.shape)
outputs= net(org_images)
output=outputs.to(device)
print(outputs.shape)
_, predicted = torch.max(outputs.data, 1)


# %%
print(outputs)

# %%
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(20,20))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


samples = []
samples_labels = []
samples_pred = []
selected = [3,66,67,0,26,16,4,13,1,11]

for i in selected:
  samples.append(org_images[i])
  samples_labels.append(org_labels[i])
  samples_pred.append(outputs[i])
samples = torch.stack(samples)
samples_labels = torch.stack(samples_labels)
samples_pred = torch.stack(samples_pred)
imshow(torchvision.utils.make_grid(samples.cpu()))
print(samples_labels)
print(samples_pred)

# %% [markdown]
# ###FGSM attack function
# In the FGSM attack, we make adversarial examples using this equation:
# $x_{adv}=x_{benign}+\epsilon * sign(\nabla_{x_{benign}}l(\theta, x, y))$

# %%
def FGSM(net, x, y, eps):
        '''
        inputs:
            net: the network through which we pass the inputs
            x: the original example which we aim to perturb to make an adversarial example
            y: the true label of x
            eps: perturbation budget

        outputs:
            x_adv : the adversarial example constructed from x
            h_adv: output of the last softmax layer when applying net on x_adv
            y_adv: predicted label for x_adv
            pert: perturbation applied to x (x_adv - x)
        '''

        x_ = Variable(x.data, requires_grad=True)
        h_ = net(x_)
        criterion= torch.nn.CrossEntropyLoss()
        cost = criterion(h_, y)
        net.zero_grad()
        cost.backward()

        #perturbation
        pert= eps*x_.grad.detach().sign()

        x_adv = x_ + pert

        h_adv = net(x_adv)
        _,y_adv=torch.max(h_adv.data,1)
        return x_adv, h_adv, y_adv, pert


# %% [markdown]
# ### Creating adversarial examples from samples with the FGSM attack and eps = 1/255

# %%
print()
print('from left to right: (1/eps) perturbation, original image, adversarial example')
print()
for i in selected:
    eps=1/255
    while True:
        x_adv, h_adv, y_adv, pert=FGSM(net, org_images[i].unsqueeze_(0),org_labels[i].unsqueeze_(0),eps)
        if y_adv.item()==org_labels[i].item():
            eps=eps+(1/255)
        else:
            break
    print("true label:", org_labels[i].item(), "adversary label:", y_adv.item())
    triple=[]
    with torch.no_grad():
        triple.append((1/eps)*pert.detach().clone().squeeze_(0))
        triple.append(org_images[i])
        triple.append(x_adv.detach().clone().squeeze_(0))
        triple=torch.stack(triple)
        grid = torchvision.utils.make_grid(triple.cpu()/2+0.5)
        plt.figure(figsize=(10,10))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.axis('off')
        plt.show()



# %% [markdown]
# **As you can see, the original and adversarial examples look extremely similar to the human eye.**

# %% [markdown]
# ### Adversarial Training with FGSM

# %% [markdown]
# First, we should build a new model (which we call net_adv) to train on adversarial examples generated by the FGSM attack.

# %%
print('==> Building new model..')
net_adv = ResNet18()
net_adv = net_adv.to(device)
if device == 'cuda':
    net_adv = torch.nn.DataParallel(net_adv)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer_adv = optim.SGD(net_adv.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler_adv = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adv, T_max=200)


# %% [markdown]
# Train_adv function trains a given neural network on adversarial examples generated from training data using the FGSM attack.
# 
# 

# %%
def train_adv(epoch, net):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    eps=8/255
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs_ = Variable(inputs.data, requires_grad=True)
        h_ = net(inputs_)

        cost = criterion(h_, targets)

        net.zero_grad()
        cost.backward()

        pert= eps*inputs_.grad.detach().sign()
        x_adv = inputs_ + pert

        optimizer_adv.zero_grad()
        outputs = net(x_adv)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_adv.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss/len(trainloader)

# %%
train_losses_adv=[]
test_losses_adv=[]
epochs=15

for epoch in range(0,epochs):
    train_losses_adv.append(train_adv(epoch, net_adv))
    test_losses_adv.append(test(epoch, net_adv))
    scheduler_adv.step()
    print(epoch)

# %%
print('Accuracy of the network on unperturbed test images: %d %%' % (acc))

# %% [markdown]
# #### Comparing naturally-trained and adversarially-trained models

# %% [markdown]
# Train - natural: training loss of the naturally-trained model
# 
# Train - adversary: training loss of the adversarially-trained model
# 
# Test - natural:  loss of the naturally-trained model on original (unperturbed) test images
# 
# Test - adversary: loss of the adversarially-trained model on original (unperturbed) test images

# %%
plt.plot(np.arange(1,epochs+1),train_losses, label='train - natural')
plt.plot(np.arange(1,epochs+1), test_losses, label='test - natural')
plt.plot(np.arange(1,epochs+1),train_losses_adv, label='train - adversary')
plt.plot(np.arange(1,epochs+1), test_losses_adv, label='test - adversary')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.legend()
plt.show()

# %% [markdown]
# Training losses of the adversarially-trained model are higher than training losses of the naturally-trained model, which is intuitive since the adversarially-trained model is trained against adversarial examples, which makes it harder for the model to label these perturbed inputs correctly and results in higher errors.
# 
# The loss of the naturally-trained model on test data is higher than the training loss, since test data is unseen by the model, resulting in higher error in classification.
# 
# However, the loss of the adversarially-trained model on test data is lower than the corresponding training loss. This is probably because the test instances are not adversarial (in contrast to training data) and that the model has learned to extract important and useful features, thus performing better on test images.

# %% [markdown]
# ####Evaluating the adversarially-trained model with FGSM against FGSM attack on test data
# 

# %% [markdown]
# Test_adv function constructs adversarial examples from test data (with FGSM using net) and evaluates net_adv on them.

# %%
def test_adv(net, net_adv, eps):
    accuracy=0
    net.train()
    net_adv.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        x_adv, h_adv, y_adv, pert = FGSM (net, inputs, targets, eps)

        outputs = net_adv(x_adv)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# %%
net2 = net
accuracy = test_adv(net, net2, eps)
print("accuracy of normal network (without adversarial training) on adversarially attacked dataset: ", accuracy)

# %%
for eps in [4/355, 8/255, 12/255]:
    accuracy=test_adv(net, net_adv, eps)
    print("epsilon:", eps, "accuracy:", accuracy)

# %% [markdown]
# PGD Training

# %%
# PGD Attack
# MNIST init
def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40) :
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters) :
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

# %%
def train_adv_pgd(epoch, net):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    eps=8/255
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs_ = Variable(inputs.data, requires_grad=True)
        pgd_attacked_imgs = pgd_attack(net, inputs_, targets)
        # h_ = net(inputs_)

        # cost = criterion(h_, targets)

        # net.zero_grad()
        # cost.backward()

        # pert= eps*inputs_.grad.detach().sign()
        # x_adv = inputs_ + pert



        optimizer_adv.zero_grad()
        outputs = net(pgd_attacked_imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_adv.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss/len(trainloader)

# %%
print('==> Building new model..')
net_adv2 = ResNet18()
net_adv2 = net_adv2.to(device)
if device == 'cuda':
    net_adv2 = torch.nn.DataParallel(net_adv2)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer_adv2 = optim.SGD(net_adv.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler_adv2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adv2, T_max=200)


# %%
train_losses_adv=[]
# test_losses_adv=[]
epochs=15

for epoch in range(0,epochs):
    train_losses_adv.append(train_adv_pgd(epoch, net_adv2))
    # test_losses_adv.append(test_adv(epoch, net_adv2))
    scheduler_adv.step()
    print(epoch)

# %%
for eps in [4/355, 8/255, 12/255]:
    accuracy=test_adv(net, net_adv2, eps)
    print("epsilon:", eps, "accuracy:", accuracy)

# %% [markdown]
# We see that **as epsilon increases, the accuracy decreases**, since for bigger epsilons, greater perturbations are allowed, therefore it becomes harder for the model to correctly label those perturbed examples.


