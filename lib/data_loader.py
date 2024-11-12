import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MINCDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the .txt file with image file names.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
                If None, a default transform is applied that includes converting images to tensors.
        """

        with open(txt_file, 'r') as file:
            self.image_paths = file.readlines()
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
        self.classes = sorted(os.listdir(root_dir+'/images'))  # Classes are directory names

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx].strip()
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Extract label from the image path
        label_name = img_name.split('/')[1]  # Assuming format: images/class_name/image_name.jpg
        label = self.classes.index(label_name)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loader(txt_file, root_dir, batch_size=32, shuffle=True, num_workers=4, transform=None):
    """
    Utility function to get DataLoader.
    """
    dataset = MINCDataset(txt_file, root_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

if __name__ == '__main__':
    # Example usage
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_loader = get_data_loader('/data/scratch/ec23984/data-repo/minc/minc-2500/labels/test1.txt', '/data/scratch/ec23984/data-repo/minc/minc-2500', batch_size=32, transform=transform)
    
    # Testing the data loader
    for images, labels in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break
