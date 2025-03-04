import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch
from sklearn.model_selection import train_test_split

from PvalueExplainer import PvalueExplainer

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Filter the dataset for labels 3 and 5
indices = np.where((mnist.targets == 3) | (mnist.targets == 5))[0]
filtered_mnist = torch.utils.data.Subset(mnist, indices)

# Function to display images
def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    for i in range(num_images):
        image, label = dataset[np.random.randint(len(dataset))]
        axes[i].imshow(image.squeeze(), cmap='gray')
        # axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

# Display random images
show_images(filtered_mnist)

X = []
y = []
for image, label in filtered_mnist:
    X.append(image.numpy().flatten())
    y.append(label)
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

p_values = PvalueExplainer.feat_imp_test(X_train, y_train, n_est=500, n_rep=10000, clf="SPORF")
top_features = p_values < 0.05
X_val_filtered = X_val[:, top_features]
X_test_filtered = X_test[:, top_features]


significant_features, p_values = PvalueExplainer.get_significant_features(
    X_train, y_train, 
    n_est=500, 
    n_rep=100000, 
    clf="SPORF"
)

X_val_filtered = PvalueExplainer.filter_features(X_val, significant_features)
X_test_filtered = PvalueExplainer.filter_features(X_test, significant_features)