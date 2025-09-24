from data_loader import FaceDataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225]),
])

dir = '/Volumes/T7/projects/age_detector/dataset'
dataset_ =  FaceDataset(root_dir = dir, transform = transform)
dataloader = DataLoader(dataset_, batch_size = 32, shuffle = True, num_workers = 4)

print(next(iter(dataloader)))