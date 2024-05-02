import torch
import argparse
from dataset import load_data
from model import get_model

def test_model(model, test_loader, device):
    """
    Evaluate the performance of a machine learning model on a test dataset.

    Parameters:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) to perform the test on.

    Prints:
        The accuracy of the model on the test set.
    """
    
    # Move the model to the specified device and set it to evaluation mode
    model = model.to(device)
    model.eval() 

    correct = 0
    total = 0

    # Disable gradient computation for testing, which reduces memory usage and speeds up computation
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the ML model')
    parser.add_argument('--model_path', type=str, default='./model/best_model.pth', help='Path to the trained model file')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Directory with data')

    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = get_model()
    model.load_state_dict(torch.load(args.model_path))

    # Load the test data
    _, _, test_loader = load_data(args.data_dir + '/train', args.data_dir + '/val', args.data_dir + '/test')

    # Test the model
    test_model(model, test_loader, device)
