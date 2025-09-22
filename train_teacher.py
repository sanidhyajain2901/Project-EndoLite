import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, datasets, transforms
from tqdm import tqdm
import os
import logging
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, save_dir='ResNet_checkpoint'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, verbose=True
        )
        
        logging.basicConfig(
            filename=os.path.join(save_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        model_summary = str(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        framework_versions = (
            f"PyTorch version: {torch.__version__}, "
            f"Torchvision version: {torchvision.__version__}"
        )
        
        logging.info("Model Structure:")
        logging.info(model_summary)
        logging.info(f"Total Parameters: {total_params}")
        logging.info(f"Trainable Parameters: {trainable_params}")
        logging.info(framework_versions)
        
        print("Model Structure:")
        print(model_summary)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(framework_versions)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{train_loss/(pbar.n + 1):.3f}', 
                              'acc': f'{100.*correct/total:.2f}%'})
        
        return train_loss/len(self.train_loader), 100.*correct/total

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return val_loss/len(self.val_loader), 100.*correct/total

    def train(self, num_epochs=200, val_freq=5):
        best_val_acc = 0
        history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch + 1)
            
            val_loss, val_acc = 0, 0
            if (epoch + 1) % val_freq == 0:
                val_loss, val_acc = self.validate()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(epoch, val_acc, is_best=True)
                self.scheduler.step(val_acc)

            log_msg = (f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, '
                       f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, '
                       f'Val Acc: {val_acc:.2f}%')
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            logging.info(log_msg)
            print(log_msg)
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc)
        
        return history

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'model_best.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            
        torch.save(checkpoint, path)

def plot_metrics(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    print(f"Training metrics plot saved to {plot_path}")
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'Dataset'
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    trainer = Trainer(model, train_loader, val_loader, device)
    history = trainer.train(num_epochs=200, val_freq=1)
    
    if history:
        plot_metrics(history, trainer.save_dir)

if __name__ == '__main__':
    print("Starting model training...")
    try:
        main()
        print("Training completed successfully!")
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")