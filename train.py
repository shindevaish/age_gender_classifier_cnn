import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss, CrossEntropyLoss
from torchvision import transforms
import torch.optim as optim
import os

from data_loader import FaceDataset
from model import AgeGenderModel


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  

    full_dataset = FaceDataset(root_dir='/Users/vaishnavishinde/Desktop/age_detector/dataset', transform=transform)

    filtered_indices = [i for i, (data, label) in enumerate(full_dataset) if data is not None]
    dataset = torch.utils.data.Subset(full_dataset, fitered_indices)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_words = 4)
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False, num_words = 4)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False, num_words = 4)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = AgeGenderModel().to(device)

    age_criterion = MSELoss()
    gender_criterion = CrossEntropyLoss()
    optimizer - optim.Adam(model.parameters(), lr = 0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader :
            images = images.to(device)
            ages = labels[:, 0].to(device)
            genders = labels[:, 1].long().to(device)

            optimizer.zero_grad()

            age_pred, gender_pred = model(images)

            age_loss = age_criterion(age_pred.squeeze(), ages)
            gender_loss = gender_criterion(gender_pred, genders)
            total_loss = age_loss + gender_loss

            total_loss.backward()

            optimizer.step()

            running_loss += total_loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):4f}")


