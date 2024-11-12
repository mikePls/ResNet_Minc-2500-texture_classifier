import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.data_loader import get_data_loader
from core.network import resnet18


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,scheduler, device, num_epochs=500, lr_scheduler=None, checkpoint_dir='checkpoints', log_dir='logs'):
        """
        Initialize the Trainer class.

        Args:
            model (torch.nn.Module): Model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            device (torch.device): Device to run the model on (CPU or GPU).
            num_epochs (int): Number of training epochs.
            lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
            checkpoint_dir (str): Directory to save model checkpoints.
            log_dir (str): Directory to save TensorBoard logs.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.writer = SummaryWriter(log_dir=log_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train(self):
        """Main training loop."""
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss, train_correct = 0, 0
            running_loss = 0.0

            # Initialize tqdm progress bar
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}', leave=False)
            
            for i, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels.data)
                
                # Update running loss and progress bar
                running_loss += loss.item()
                progress_bar.set_postfix(running_loss=running_loss/(i+1))

            train_loss /= len(self.train_loader.dataset)
            train_acc = train_correct.double() / len(self.train_loader.dataset)

            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)

            # Validation phase
            val_loss, val_acc = self.validate(epoch)

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Logging
            print(f'Epoch {epoch+1}/{self.num_epochs} - '
                  f'Train loss: {train_loss:.4f} - Train acc: {train_acc:.4f} - '
                  f'Val loss: {val_loss:.4f} - Val acc: {val_acc:.4f}')

            if epoch % 100 == 0 and epoch >=100:
                self.save_checkpoint(epoch, val_loss)

    def validate(self, epoch):
        """Validation loop."""
        self.model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_loss /= len(self.val_loader.dataset)
        val_acc = val_correct.double() / len(self.val_loader.dataset)

        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/val', val_acc, epoch)

        return val_loss, val_acc

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader = get_data_loader('/data/scratch/ec23984/data-repo/minc/minc-2500/labels/train1.txt', '/data/scratch/ec23984/data-repo/minc/minc-2500', batch_size=32, transform=None)
    val_loader = get_data_loader('/data/scratch/ec23984/data-repo/minc/minc-2500/labels/validate1.txt', '/data/scratch/ec23984/data-repo/minc/minc-2500', batch_size=32, transform=None)

  
    model = resnet18(num_classes=23)  # 23 classes for Minc-2500
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)


    # Initialize Trainer and start training
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, 
                      criterion=criterion, optimizer=optimizer,scheduler=scheduler,
                        device=device, num_epochs=450)
    trainer.train()