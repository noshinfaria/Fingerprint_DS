import torch
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# MoCoV3 model definition
class MoCoV3(nn.Module):
    def __init__(self, base_encoder=models.resnet50, dim=256, mlp_dim=4096, T=0.2, momentum=0.999):
        super(MoCoV3, self).__init__()

        self.T = T
        self.momentum = momentum

        # Create the encoders (backbone + projection MLP)
        self.encoder_q = base_encoder(pretrained=False)
        self.encoder_k = base_encoder(pretrained=False)

        # Get the number of features in the last layer of ResNet
        in_features = self.encoder_q.fc.in_features # fully connected layer

        # MLP projection head for query and key encoders
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(in_features, mlp_dim),
            nn.GELU(), # Gaussian Error Linear Unit; non-linear activation function
            # GELU tends to work well in transformer models and other deep architectures, making it a popular choice over ReLU or sigmoid activations.
            nn.Linear(mlp_dim, dim) #  convert it in smaller feature dimension to use in contrastive learning frameworks(MoCo v3)
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(in_features, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

        # Initialize key encoder's parameters to match query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Stop gradient for key encoder

    @torch.no_grad() # No gradients will be tracked inside this function
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    def forward(self, im_q, im_k):
        # Query feature extraction
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # Key feature extraction (momentum updated encoder)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        return q, k

# Contrastive loss (InfoNCE loss)
def contrastive_loss(q, k):
    N = q.size(0)

    # Positive logits: dot product between q and corresponding k
    pos_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) #einsum = Einstein summation convention
    #The .unsqueeze(-1) method call adds a new dimension at the end of the tensor, changing the shape from (N,) to (N, 1). 
    #This is often done to ensure that the tensor has the right shape for subsequent operations, especially in loss calculations where broadcasting is needed

    # Negative logits: dot product between q and all k in the batch
    neg_logits = torch.einsum('nc,mc->nm', [q, k]) # m corresponds to a potentially different batch size

    # Concatenate positive and negative logits
    logits = torch.cat([pos_logits, neg_logits], dim=1)

    # Apply temperature scaling
    logits /= 0.2  # Temperature (T)

    # Labels: positive key is the first in the list of logits
    labels = torch.zeros(N, dtype=torch.long).to(q.device)

    # Cross entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss

# Training loop
def train_moco_v3(model, data_loader, optimizer, device, epochs=10):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for idx, (images) in enumerate(data_loader):
            # Simulate two different views for query and key by applying different augmentations
            im_q = augmentation(images).to(device)  # Augmentation 1 (Query)
            print(f"Augmented Query Image {idx + 1}")
            im_k = augmentation(images).to(device)  # Augmentation 2 (Key)
            print(f"Augmented Key Image {idx + 1}")

            # Forward pass through MoCoV3
            q, k = model(im_q, im_k)

            # Compute contrastive loss
            loss = contrastive_loss(q, k)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss per epoch
        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("Training complete")
    torch.save(model.state_dict(), '/home/noshin/JunkBox/fingerprint/fingerprint_DS/moco_v3_pretrained.pth')
    print("Model saved")


# Simple augmentation for query/key views (MoCo v3 typically uses stronger augmentations)
def augmentation(image_batch):
    aug_transform = transforms.Compose([
         transforms.Resize((224, 224)),  # Ensure all images are resized to 224x224
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return aug_transform(image_batch)


class SOCOFingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('BMP')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))  # Ensure all images are resized to 224x224

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  # Convert to tensor if no transform is provided


        return image  # 0 can be a placeholder for labels if not needed

class FVC2000Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('tif')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))  # Ensure all images are resized to 224x224

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  # Convert to tensor if no transform is provided


        return image  # 0 can be a placeholder for labels if not needed


# Set up data loader 
def get_data_loader(socofing_dir, fvc_dir, batch_size=32):
    socofing_dataset = SOCOFingDataset(root_dir=socofing_dir, transform=None)
    fvc2000_dataset = FVC2000Dataset(root_dir=fvc_dir, transform=None)

    combined_dataset = torch.utils.data.ConcatDataset([socofing_dataset, fvc2000_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    return train_loader


if __name__ == "__main__":
    # Check if GPU is available, else fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and optimizer
    model = MoCoV3(base_encoder=models.resnet50).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4) # SGD= Stochastic Gradient Descent
    #momentum = add fraction leading to fast convergence; Weight Decay= regularization technique to prevent overfitting, add penalty

    # Set up data loader
    socofing_dir = '/home/noshin/JunkBox/fingerprint/fingerprint_DS/SOCOFing'  # Set the correct path to your SOCOFing dataset
    fvc_dir = '/home/noshin/JunkBox/fingerprint/fingerprint_DS/merged_FVC'
    data_loader = get_data_loader(socofing_dir, fvc_dir)

    # Train the model
    train_moco_v3(model, data_loader, optimizer, device, epochs=10)

