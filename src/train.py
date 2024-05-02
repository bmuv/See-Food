import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import load_data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from model import get_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=10, early_stopping_rounds=5, start_epoch=0):
    """
    Train a PyTorch model with given parameters.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        scheduler (torch.optim.lr_scheduler): Scheduler to adjust learning rates.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        num_epochs (int): Total number of epochs to train.
        early_stopping_rounds (int): Number of epochs to stop training if no improvement.
        start_epoch (int): Epoch number to start/resume training.

    Outputs:
        Model training process with periodic validation and possibility of early stopping.
    """
    
    model = model.to(device)
    best_val_loss = float('inf')
    no_improve_epochs = 0

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct, total = 0, 0

        # Iterate over batches of data
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate training loss and accuracy
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_accuracy = 100 * correct / total

                tepoch.set_postfix(loss=loss.item(), accuracy=train_accuracy)

        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        all_targets = []
        all_outputs = []

        # Process validation data
        with torch.no_grad(), tqdm(val_loader, unit="batch") as vepoch:
            for images, labels in vepoch:
                vepoch.set_description(f"Validation Epoch {epoch+1}")

                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_targets.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_outputs.extend(probs)

                vepoch.set_postfix(loss=loss.item())

        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_roc_auc = roc_auc_score(all_targets, all_outputs)
        val_accuracy = accuracy_score(all_targets, (np.array(all_outputs) > 0.5).astype(int)) * 100

        # Log validation results
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation ROC-AUC: {val_roc_auc:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(avg_val_loss)  # Adjust the learning rate

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), f'./model/best_model.pth')
            print(f'Model saved: improved validation loss to {best_val_loss:.4f}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_rounds:
                print(f'Early stopping triggered after {early_stopping_rounds} epochs with no improvement')
                break

if __name__ == "__main__":
    # Load dataset using custom DataLoader
    train_dir = './dataset/train' 
    test_dir = './dataset/test'   
    val_dir = './dataset/val'   
    train_loader, val_loader, test_loader = load_data(train_dir, val_dir, test_dir)
    
    # Check if there's a saved model
    model_path = './model/best_model_.pth'
    if os.path.exists(model_path):
        model = get_model()
        model.load_state_dict(torch.load(model_path))
        start_epoch = 0 # Load the epoch number where you left off
    else:
        model = get_model()
        start_epoch = 0
    
    # Setup loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # Added weight decay
    
    # Optional: Load optimizer state if resuming training
    # optimizer_path = './model/optimizer_state.pth'
    # if os.path.exists(optimizer_path):
    #     optimizer.load_state_dict(torch.load(optimizer_path))
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) # Learning rate scheduler
    
    # Start training process
    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=50, early_stopping_rounds=20, start_epoch=start_epoch)
