import torch.nn as nn
import torch.optim as optim

from vit_base_cls import VisionTransformer
from vit_base_cls import device
from dataset import train_loader, test_loader
import torch

if __name__ == "__main__":

    # Initialize the model
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        embed_dim=4,
        depth=1,
        n_heads=1,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.1,
        attn_p=0.1,
        num_classes=10,  # CIFAR-10 has 10 classes
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(10):  # Train for 10 epochs
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")



    # Evaluation loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")