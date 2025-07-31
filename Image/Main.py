import torch

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split, DataLoader, Subset
from tqdm import tqdm
import os
import sys
import time
import random
import numpy as np
from TrainModel import TrainModel

from CNNetworks import NIN_MNIST, NIN_CIFAR10, NIN_SVHN, NIN_EMNIST, NIN, VGG, CNN_USPS, Food101Net, VGG_office31
from RunGurobi import Gurobi_STC, Gurobi_CmC_Any, Gurobi_CmC_Correct

def GetModel(dataset_name, num_classes=10, device=None):
    if dataset_name == "MNIST":
        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    elif dataset_name == "CIFAR10":
        model_t = VGG(num_classes=10).to(device)
        model_g = VGG(num_classes=10).to(device)
    elif dataset_name == "FashionMNIST":
        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    elif dataset_name == "KMNIST":
        model_t = NIN_MNIST(num_classes=10).to(device)
        model_g = NIN_MNIST(num_classes=10).to(device)
    elif dataset_name == "EMNIST":
        model_t = NIN_EMNIST(num_classes=26).to(device)
        model_g = NIN_EMNIST(num_classes=26).to(device)
    elif dataset_name == "SVHN":
        model_t = VGG(num_classes=10).to(device)
        model_g = VGG(num_classes=10).to(device)
    elif dataset_name == "PathMNIST":
        model_t = VGG(num_classes=9).to(device)
        model_g = VGG(num_classes=9).to(device)
    elif dataset_name == "Food101":
            model_t = VGG(num_classes=10).to(device)
            model_g = VGG(num_classes=10).to(device)
    elif dataset_name == "USPS":
        model_t = CNN_USPS(num_classes=10).to(device)
        model_g = CNN_USPS(num_classes=10).to(device)
    elif dataset_name == "Caltech101":
        model_t = VGG(num_classes=101).to(device)
        model_g = VGG(num_classes=101).to(device)
    elif dataset_name == "office31":
        model_t = VGG_office31(num_classes=31).to(device)
        model_g = VGG_office31(num_classes=31).to(device)

    return model_t, model_g

def GetDataset(dataset_name, root_dir='./data', device=None):
    if dataset_name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "KMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    elif dataset_name == "EMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)
    
    elif dataset_name == "SVHN":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])])
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    
    elif dataset_name == "Food101":
        transform = transforms.Compose([transforms.Resize((64, 64)),transforms.ToTensor()])
        train_dataset = torchvision.datasets.Food101(root="./data", split="train", download=True, transform=transform)
        test_dataset = torchvision.datasets.Food101(root="./data", split="test", download=True, transform=transform)
        all_classes = train_dataset.classes
        random.seed(42)
        selected_classes = random.sample(all_classes, 10)

        def fast_filter_food101(dataset, selected_classes):
            class_names = dataset.classes
            label_names = [class_names[label] for label in dataset._labels]
            selected_indices = [i for i, name in enumerate(label_names) if name in selected_classes]
            return Subset(dataset, selected_indices)

        class_map = {cls_name: i for i, cls_name in enumerate(selected_classes)}
        train_filtered = fast_filter_food101(train_dataset, selected_classes)
        test_filtered = fast_filter_food101(test_dataset, selected_classes)

        train_dataset = RelabelSubset(train_filtered, class_map, train_dataset)
        test_dataset = RelabelSubset(test_filtered, class_map, test_dataset)


    elif dataset_name == "USPS":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.USPS(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.USPS(root='./data', train=False, download=True, transform=transform)

    elif dataset_name == "Caltech101":
        train_dataset, test_dataset = get_loaders_from_folder("./data/caltech-101/101_ObjectCategories", image_size=(64, 64), val_split=0.25)  

    elif dataset_name == "office31":
        train_dataset, test_dataset = get_loaders_from_folder("./data/office31/amazon", image_size=(64, 64), val_split=0.2)

    return train_dataset, test_dataset   

class RelabelSubset(torch.utils.data.Dataset):
    def __init__(self, subset, class_map, orig_dataset):
        self.subset = subset
        self.class_map = class_map
        self.orig_dataset = orig_dataset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        class_name = self.orig_dataset.classes[label]
        new_label = self.class_map[class_name]
        return image, new_label

def get_loaders_from_folder(root_dir, image_size=(224, 224), val_split=0.2, seed=42):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    full_dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    return train_dataset, val_dataset


if __name__ == "__main__":
    os.makedirs("Stats/STC", exist_ok=True)
    os.makedirs("Stats/CmC", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initEpoch = 300
    G_epoch = 100
    n_samples_gurobi = -1
    optimize = "Adam"

    dataset_name = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else "STC"
    misc_type = sys.argv[3] if len(sys.argv) > 3 else "A"
    misclassification_count = sys.argv[4] if len(sys.argv) > 4 else "1"  
    
    if method == "CmC":
        n_samples_gurobi = 1000

    train_dataset, test_dataset = GetDataset(dataset_name)

    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_size = len(train_dataset)
    val_size = len(test_dataset)
    total_size = train_size + val_size
    
    model_t, model_g = GetModel(dataset_name, device=device)

    rng = np.random.default_rng(seed=42)
    all_indices = rng.permutation(total_size)

    new_train_indices = all_indices[:train_size]
    new_val_indices = all_indices[train_size:]

    train_subset = Subset(full_dataset, new_train_indices)
    val_subset = Subset(full_dataset, new_val_indices)

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    learningRate = 0.01
    
    if os.path.exists(f"./checkpoints/{dataset_name}/Run1_full_checkpoint.pth") == False:
        TM = TrainModel(method, dataset_name, model_t, train_loader, val_loader, device, num_epochs=initEpoch, resume_epochs=G_epoch, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="Train", run_id=1, start_experiment=True)
        try:
            TM.run()
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)

    TM_after_g = TrainModel(method, dataset_name, model_g, train_loader, val_loader, device, num_epochs=G_epoch, resume_epochs=0, batch_size=64, learning_rate=learningRate, optimizer_type=optimize, phase="GurobiEdit", run_id=1)

    if device.type == 'cuda':
        checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run1_full_checkpoint.pth")
    else:
        checkpoint = torch.load(f"./checkpoints/{dataset_name}/Run1_full_checkpoint.pth", map_location=torch.device('cpu'))
    TM_after_g.model.load_state_dict(checkpoint['model_state_dict'])
    TM_after_g.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    TM_after_g.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Loaded model from checkpoint.")
    TM_after_g.save_fc_inputs("Train")
    TM_after_g.save_fc_inputs("Val")
    print(f"Saved FC inputs.")
    if method == "STC":
        Gurobi_output = Gurobi_STC(dataset_name, TM_after_g.log_file, 1, n=n_samples_gurobi)
    elif method == "CmC":
        if misc_type == "A":
            Gurobi_output = Gurobi_CmC_Any(dataset_name, TM_after_g.log_file, 1, n=n_samples_gurobi, misclassification_count=misclassification_count)
        elif misc_type == "C":
            Gurobi_output = Gurobi_CmC_Correct(dataset_name, TM_after_g.log_file, 1, n=n_samples_gurobi, misclassification_count=misclassification_count)
    if Gurobi_output is None:
        print("Gurobi did not find a solution.")
        sys.exit(1)
    W2_new, b2_new = Gurobi_output
    TM_after_g.delete_fc_inputs()
    new_W = torch.tensor(W2_new).to(model_g.classifier.weight.device)
    new_b = torch.tensor(b2_new).to(model_g.classifier.bias.device)
    with torch.no_grad():
        TM_after_g.model.classifier.weight.copy_(new_W)
        TM_after_g.model.classifier.bias.copy_(new_b)
    train_loss, train_acc = TM_after_g.evaluate("Train")
    val_loss, val_acc = TM_after_g.evaluate("Val")

    with open(TM_after_g.log_file, "a") as f:
        f.write(f"1,Gurobi_Complete_Eval_Train,-1,{train_loss},{train_acc}\n")
        f.write(f"1,Gurobi_Complete_Eval_Val,-1,{val_loss},{val_acc}\n")
    try:
        TM_after_g.run()
    except Exception as e:
        print(f"Error during training: {e}")
        

