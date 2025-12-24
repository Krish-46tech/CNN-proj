import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image
import numpy as np
import cv2
import os


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)


def get_dataloaders(data_dir, batch_size=16):
    # Augmentation for train
    train_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir,"train"), train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir,"val"), val_tfms)

    # Compute class weights for WeightedRandomSampler
    class_counts = [0]*len(train_ds.classes)
    for _, label in train_ds.samples:
        class_counts[label] += 1
    class_weights = [sum(class_counts)/c for c in class_counts]
    sample_weights = [class_weights[label] for _, label in train_ds.samples]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, epochs=10):
    # Compute class weights for loss
    class_counts = [0]*2
    for _, label in train_loader.dataset.samples:
        class_counts[label] += 1
    weights = torch.tensor([class_counts[1]/sum(class_counts), class_counts[0]/sum(class_counts)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        st.write(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), "pneumonia_model.pth")
    st.write("Model training finished and saved!")


def predict_image(model, img_pil):
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = tfm(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
    return pred, probs


def generate_gradcam(model, img_pil):
    model.eval()
    gradients, activations = None, None

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out

    target_layer = model.model.layer4
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = np.array(img_pil)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    input_tensor = tfm(img_pil).unsqueeze(0).to(device)

    output = model(input_tensor)
    class_idx = output.argmax()
    model.zero_grad()
    output[0, class_idx].backward()

    pooled_grads = torch.mean(gradients, dim=[0,2,3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap /= heatmap.max()

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return result


st.title("Pneumonia Detection with Grad-CAM")


uploaded_file = st.file_uploader("Upload a chest X-ray", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
    model.eval()

  
    pred, probs = predict_image(model, img)
    st.write("Prediction:", "Normal" if pred==0 else "Pneumonia")
    st.write("Confidence: Normal {:.2f}% | Pneumonia {:.2f}%".format(probs[0]*100, probs[1]*100))

   
    if st.button("Show Grad-CAM"):
        cam_img = generate_gradcam(model, img)
        st.image(cam_img, caption="Grad-CAM", use_column_width=True)


if st.checkbox("Retrain model (requires dataset in 'data/' folder)"):
    train_loader, val_loader = get_dataloaders("data")
    model = PneumoniaCNN().to(device)
    train_model(model, train_loader, epochs=5)  
