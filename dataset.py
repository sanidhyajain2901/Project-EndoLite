import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torchvision
import logging
from tqdm import tqdm
import os

class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetV2Student, self).__init__()
        print(f"INFO: Using placeholder MobileNetV2Student model with {num_classes} classes.")
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)



class DistillationTrainer:
    """Conducts the knowledge distillation training process."""
    def __init__(self, teacher_model, student_model, train_loader, val_loader, 
                 device, temperature=4.0, alpha=0.5, save_dir='Student_checkpoint'):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.criterion_ce = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(
            self.student.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, verbose=True
        )
        
        self._setup_logging()

    def _setup_logging(self):
        """Configures logging for the training process."""
        log_path = os.path.join(self.save_dir, 'distillation.log')
        logging.basicConfig(
            filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s'
        )

        model_summary = str(self.student)
        total_params = sum(p.numel() for p in self.student.parameters())
        trainable_params = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        framework_versions = (
            f"PyTorch version: {torch.__version__}, "
            f"Torchvision version: {torchvision.__version__}"
        )
        
        header = "--- Model And Framework Summary ---"
        print(header)
        logging.info(header)
        
        print("Model Structure:")
        logging.info("Model Structure:\n" + model_summary)
        
        param_info = (f"Total Parameters: {total_params}\n"
                      f"Trainable Parameters: {trainable_params}\n"
                      f"{framework_versions}")
        print(param_info)
        logging.info(param_info)
        print("-" * len(header))

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Calculates the total distillation loss."""
       
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature * self.temperature)

        hard_loss = self.criterion_ce(student_logits, labels)
        
        l1_norm = sum(p.abs().sum() for p in self.student.parameters())
        lambda_l1 = 1e-5
        
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss + lambda_l1 * l1_norm
        return loss, l1_norm.item()
        
    def train_epoch(self, epoch):
        """Runs a single training epoch."""
        self.teacher.eval()  
        self.student.train()
        train_loss, total_l1_norm, correct, total = 0, 0, 0, 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            with torch.no_grad():
                teacher_logits = self.teacher(images)
                
            self.optimizer.zero_grad()
            student_logits = self.student(images)
            
            loss, l1_norm = self.distillation_loss(student_logits, teacher_logits, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            total_l1_norm += l1_norm
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss / (pbar.n + 1):.3f}', 
                'acc': f'{100. * correct / total:.2f}%'
            })
            
        avg_l1_norm = total_l1_norm / len(self.train_loader)
        return train_loss / len(self.train_loader), 100. * correct / total, avg_l1_norm
        
    def validate(self):
        """Validates the model on the validation set."""
        self.student.eval()
        val_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.student(images)
                loss = self.criterion_ce(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return val_loss / len(self.val_loader), 100. * correct / total
        
    def train(self, num_epochs=200, val_freq=1):
        """The main training loop."""
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc, avg_l1_norm = self.train_epoch(epoch + 1)
            
            log_msg = (f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, '
                       f'Train Acc: {train_acc:.2f}%, Avg L1 Norm: {avg_l1_norm:.4f}')
            
            val_acc = 0
            if (epoch + 1) % val_freq == 0:
                val_loss, val_acc = self.validate()
                log_msg += f', Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_checkpoint(epoch, val_acc, is_best=True)
                    
                self.scheduler.step(val_acc)
            
            logging.info(log_msg)
            print(log_msg)
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc)
                
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Saves a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc
        }
        
        filename = 'student_best.pth' if is_best else f'student_checkpoint_epoch_{epoch+1}.pth'
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)

        if is_best:
            print(f"Saved new best model to {path} with accuracy {val_acc:.2f}%")



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    # Normalization values are standard for models pre-trained on ImageNet
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

    # Data Loading using ImageFolder 
    data_dir = 'Dataset'
    print(f"Loading data from '{data_dir}'...")
    # Assumes your data is structured as Dataset/train/... and Dataset/val/...
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    #  number of classes automatically from the dataset folders
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes: {train_dataset.classes}")
    
    #Teacher Model Setup 
    teacher = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    teacher.fc = nn.Linear(teacher.fc.in_features, num_classes)
    
    try:
        teacher_checkpoint_path = 'Teacher/Teacher_Resnet_checkpoint/model_best.pth'
        teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
        teacher.load_state_dict(teacher_checkpoint['model_state_dict'])
        print(f"Successfully loaded teacher checkpoint from {teacher_checkpoint_path}")
    except FileNotFoundError:
        print(f"WARNING: Teacher checkpoint not found at {teacher_checkpoint_path}. Using pre-trained ResNet18 weights without fine-tuning.")
    except KeyError:
        teacher.load_state_dict(teacher_checkpoint)
        print(f"Successfully loaded teacher checkpoint from {teacher_checkpoint_path} (direct state_dict).")

    teacher.eval()  
    
    #Student Model Setup 
    student = MobileNetV2Student(num_classes=num_classes)
    
    #Trainer Initialization and Execution 
    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        temperature=4.0,  
        alpha=0.5
    )
    
    print("\nStarting knowledge distillation training...")
    trainer.train(num_epochs=200, val_freq=1)
    print("Knowledge distillation completed!")

if __name__ == '__main__':
    main()