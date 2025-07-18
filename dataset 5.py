import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        for label, folder in enumerate(["NORMAL", "PNEUMONIA"]):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            img = Image.open(img_path).convert("L")  # Force grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new("L", (128, 128))  # Dummy black image if error
        img = self.transform(img)  # Always apply transform
        label = self.labels[idx]
        return img, label
