import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader

from constant import *
import torchvision

class VehicleDataset(Dataset):
    def __init__(self, image_dir: str, label: int, transform=None):
        self.image_paths = []
        self.label = label
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                         torchvision.transforms.Resize(256),
                                                         torchvision.transforms.RandomCrop(224),
                                                         torchvision.transforms.RandomHorizontalFlip(),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.485, 0.456, -.406],
                                                                                          [0.229, 0.224, 0.225])
                                                         ])

        head_dir = os.path.join(DATA_DIR, image_dir)
        for d in os.listdir(head_dir):
            for suffix in ["*.png", "*.jpg"]:
                dir_path = os.path.join(head_dir, str(d), suffix)
                self.image_paths.extend(glob.glob(dir_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            self.transform(image)

        return image, self.label


if __name__ == '__main__':
    vehicle_dataset = VehicleDataset(VEHICLES_DATA_DIR, label=1)
    data_loader = DataLoader(vehicle_dataset, batch_size=10, shuffle=True)
    for data in data_loader:
        images, labels = data
        break
