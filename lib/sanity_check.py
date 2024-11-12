import os
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from data_loader import get_data_loader
import torchvision.transforms as transforms


class SanityCheck:
    def __init__(self, data_loader, classes, output_dir='sanity_check', num_samples=1):
        """
        Args:
            data_loader (DataLoader): DataLoader object to fetch data from.
            classes (list): List of class names.
            output_dir (string): Directory to save the output images.
            num_samples (int): Number of samples to show from each class (set to 1).
        """
        self.data_loader = data_loader
        self.classes = classes
        self.output_dir = output_dir
        self.num_samples = num_samples

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def select_one_per_class(self):
        """Select one image per class from the data_loader."""
        images = []
        labels = []
        selected_classes = set()
        
        for batch_images, batch_labels in self.data_loader:
            for img, label in zip(batch_images, batch_labels):
                if label.item() not in selected_classes:
                    images.append(img)
                    labels.append(label.item())
                    selected_classes.add(label.item())
                if len(selected_classes) == len(self.classes):
                    return images, labels

        return images, labels

    def create_grid(self, images, labels):
        """Creates a grid of images and adds labels as titles."""
        grid = vutils.make_grid(images, nrow=len(images), padding=2, normalize=True)

        plt.figure(figsize=(20, 10))
        plt.axis("off")
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())

        # Calculate width of each image in the grid
        img_width = grid.size(2) // len(images)
        
        # Add labels below each image
        for i, label in enumerate(labels):
            plt.text(
                i * img_width + img_width // 2,  # Center text
                grid.size(1) + 11,               # Place below the image
                self.classes[label],
                color='white',
                fontsize=10,
                ha='center',
                bbox=dict(facecolor='black', alpha=0.8)
            )

    def save_sample_grid(self, filename='sanity_check.png'):
        """Fetches one image per class, creates a grid, and saves it."""
        # Select one image per class
        images, labels = self.select_one_per_class()

        # Create and save the grid of images
        self.create_grid(images, labels)
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f'Sanity check grid saved to {save_path}')

if __name__ == '__main__':

    # Define the transform and DataLoader
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data_loader = get_data_loader('/data/scratch/ec23984/data-repo/minc/minc-2500/labels/test1.txt', '/data/scratch/ec23984/data-repo/minc/minc-2500', batch_size=32, transform=transform)

    # Define class names
    classes = ['brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair', 'leather', 'metal', 
               'mirror', 'other', 'painted', 'paper', 'plastic', 'polishedstone', 'skin', 'sky', 'stone', 
               'tile', 'water', 'wood', 'wallpaper']

    # Create a SanityCheck object and save a grid
    sanity_check = SanityCheck(data_loader, classes)
    sanity_check.save_sample_grid()
