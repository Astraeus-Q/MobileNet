import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from mobilenet import MobileNet
from utils import plot_loss_acc
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

def get_train_valid_loader(dataset_dir, batch_size, download, seed, save_images):
    dataset = torchvision.datasets.CIFAR100(
        root=dataset_dir, train=True, download=download)
    
    # Split the dataset 50000 into training set 40000 and validation set 10000.
    # print(len(dataset)) # 50000
    torch.manual_seed(seed)
    train_size = 40000
    valid_size = 10000
    train_set, valid_set = random_split(dataset, [train_size, valid_size])
    

    categories = [0 for i in range(100)]
    mean_train = 0
    std_train = 0
    i = 0
    for image, label in train_set:
        # categories[label] += 1/len(train_set) # Proportion
        categories[label] += 1

        
        image = TF.to_tensor(image)
        # print(image)
        # Compute mean and standard deviation for each channel
        mean_train += torch.mean(image, dim=(1, 2))
        std_train += torch.std(image, dim=(1, 2))


    mean_train /= len(train_set)
    std_train /= len(train_set)
    # print(mean_train, std_train)
    

    # print(categories)
    # print(sum(categories))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean_train, std_train),
    ])  

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
    ])  

    train_set.dataset.transform = transform_train
    valid_set.dataset.transform = transform_valid

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=True, num_workers=2)


    return train_loader, valid_loader

def get_test_loader(dataset_dir, batch_size):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR100(
        root=dataset_dir, train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return testloader

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main(args):
    print("Let's Go!", flush = True)
    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # train val test
    # AI6103 students: You need to create the dataloaders youself
    train_loader, valid_loader = get_train_valid_loader(args.dataset_dir, args.batch_size, True, args.seed, save_images=args.save_images) 
    if args.test:
        # print("Test")
        test_loader = get_test_loader(args.dataset_dir, args.batch_size)

    print("Data has been loaded successfully!", flush = True)

    # model
    model = MobileNet(100)
    print(model)
    model.cuda()

    # criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)

    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch+1}/{args.epochs}", flush = True)
        training_loss = 0
        training_acc = 0
        training_samples = 0
        val_loss = 0
        val_acc = 0
        val_samples = 0
        # training
        print("Start training...", flush = True)
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()

            # mixup
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, 0.2)

            batch_size = imgs.shape[0]
            optimizer.zero_grad()
            logits = model.forward(imgs)

            # mixup
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            # loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            _, top_class = logits.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            training_loss += batch_size * loss.item()
            training_samples += batch_size

        # validation
        print("Start Validation...", flush = True)
        model.eval()
        for val_imgs, val_labels in valid_loader:
            batch_size = val_imgs.shape[0]
            val_logits = model.forward(val_imgs.cuda())
            loss = criterion(val_logits, val_labels.cuda())
            _, top_class = val_logits.topk(1, dim=1)
            equals = top_class == val_labels.cuda().view(*top_class.shape)
            val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            val_loss += batch_size * loss.item()
            val_samples += batch_size
        assert val_samples == 10000
        # update stats
        stat_training_loss.append(training_loss/training_samples)
        stat_val_loss.append(val_loss/val_samples)
        stat_training_acc.append(training_acc/training_samples)
        stat_val_acc.append(val_acc/val_samples)
        # print
        print(f"Epoch {(epoch+1):d}/{args.epochs:d}.. Learning rate: {scheduler.get_lr()[0]:.4f}.. Train loss: {(training_loss/training_samples):.4f}.. Train acc: {(training_acc/training_samples):.4f}.. Val loss: {(val_loss/val_samples):.4f}.. Val acc: {(val_acc/val_samples):.4f}", flush = True)
        # lr scheduler
        scheduler.step()
    # plot
    plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)
    # test
    if args.test:
        test_loss = 0
        test_acc = 0
        test_samples = 0
        for test_imgs, test_labels in test_loader:
            batch_size = test_imgs.shape[0]
            test_logits = model.forward(test_imgs.cuda())
            test_loss = criterion(test_logits, test_labels.cuda())
            _, top_class = test_logits.topk(1, dim=1)
            equals = top_class == test_labels.cuda().view(*top_class.shape)
            test_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            test_loss += batch_size * test_loss.item()
            test_samples += batch_size
        assert test_samples == 10000
        print('Test loss: ', test_loss/test_samples)
        print('Test acc: ', test_acc/test_samples)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset_dir',type=str, help='')
    parser.add_argument('--batch_size',type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--lr',type=float, help='')
    parser.add_argument('--wd',type=float, help='')
    parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.set_defaults(lr_scheduler=False)
    parser.add_argument('--mixup', action='store_true')
    parser.set_defaults(mixup=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--save_images', action='store_true')
    parser.set_defaults(save_images=False)
    parser.add_argument('--seed', type=int, default=0, help='')
    args = parser.parse_args()
    print(args)
    main(args)
