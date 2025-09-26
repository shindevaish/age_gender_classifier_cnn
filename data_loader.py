import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir , transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        try:
            parts = img_name.split('_')
            age = int(parts[0])
            gender = int(parts[1])
        except (IndexError, ValueError):
            print(f"ERROR parsing filename: {img_name}")
            return None, None, None
        
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor([age, gender], dtype = torch.float32)
    
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225]),
])

dir = 'dataset'
dataset_ =  FaceDataset(root_dir = dir, transform = transform)
dataloader = DataLoader(dataset_, batch_size = 32, shuffle = True, num_workers = 4)