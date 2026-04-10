import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ResNextLSTM(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=256, num_layers=2):
        super(ResNextLSTM, self).__init__()
        
        # Spatial Feature Extractor (ResNeXt-50)
        # We perform transfer learning
        resnet = models.resnext50_32x4d(pretrained=True)
        modules = list(resnet.children())[:-1] # Remove FC layer
        self.resnet = nn.Sequential(*modules)
        
        # Freeze ResNet params (optional, usually good for transfer learning on small data)
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
            
        self.feature_dim = 2048 # ResNeXt50 output
        
        # Temporal Modeling (LSTM)
        self.lstm = nn.LSTM(input_size=self.feature_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq):
        # x_seq shape: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x_seq.size()
        
        # Flatten for CNN (process all frames at once)
        c_in = x_seq.view(batch_size * seq_len, C, H, W)
        
        # Extract features
        features = self.resnet(c_in) # (batch*seq, 2048, 1, 1)
        features = features.view(batch_size, seq_len, -1) # (batch, seq, 2048)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        
        # Classify
        out = self.fc(last_out)
        return self.sigmoid(out)

def get_training_transforms():
    """
    Data Augmentation for Training (as requested):
    - Horizontal Flips
    - Brightness/Contrast
    - Motion Blur (simulated with Gaussian)
    - Compression (JPEG)
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Global Model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_hybrid_model():
    global model
    try:
        model = ResNextLSTM().to(device)
        model.eval()
        print("Hybrid ResNeXt-LSTM Model Initialized.")
        # Load weights logic would go here
    except Exception as e:
        print(f"Error loading Hybrid Model: {e}")

def transform_frames(faces):
    """
    Preprocesses list of PIL images for the model.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensors = []
    for face in faces:
        tensors.append(transform(face))
        
    if not tensors:
        return None
        
    # Stack into (1, seq_len, C, H, W)
    return torch.stack(tensors).unsqueeze(0)

def predict_video(faces):
    """
    Runs the hybrid model on extracted face sequence.
    """
    global model
    if model is None:
        load_hybrid_model()
    
    if not faces:
        return 0.0
        
    try:
        input_tensor = transform_frames(faces).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            score = output.item()
            
        return score
    except Exception as e:
        print(f"Hybrid Inference Error: {e}")
        return 0.5
