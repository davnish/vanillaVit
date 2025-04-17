from matplotlib import pyplot as plt
from dataset import train_loader


# Visualize a batch of images


# show_batch_images(*next(iter(train_loader)))



plt.imshow(train_loader.dataset[0][0].permute(1, 2, 0))
plt.title(f"Label: {train_loader.dataset[0][1]}")
plt.axis('off') # Hide axis 
plt.show()