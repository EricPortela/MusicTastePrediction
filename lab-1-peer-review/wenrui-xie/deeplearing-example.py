import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import numpy as np

#use chatgpt to fix random number seed

def set_seed(seed):
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_seed(60)

data=pd.read_csv('training_data.csv', sep=',')
test_data=pd.read_csv('songs_to_classify.csv', sep=',')

# select which features to use
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

test_data = test_data.iloc[:].values

# use chatgpt to delete outlier
Iso = IsolationForest(contamination=0.02)     
outliers = Iso.fit_predict(X) 
outlier_indices = np.where(outliers == -1)
X = np.delete(X, outlier_indices, axis=0)
y = np.delete(y, outlier_indices, axis=0)

# Normalize input data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
test_data = scaler.fit_transform(test_data)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = X
# y_train = y

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

test_data = torch.tensor(test_data, dtype=torch.float32)


# use chatgpt and changed by myself, use CNN to predict output
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 13, 128)  # Adjust to match the number of features
        self.fc2 = nn.Linear(128, 1)  # Binary output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Instantiate model, define loss function and optimizer
model = CNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 250
batch_size = 64

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_test).sum().item() / y_test.size(0)

    print(f'Accuracy: {accuracy:.4f}')
    predictions = model(test_data)
    predicted_classes = (predictions > 0.5).float()
    # print(predicted_classes.reshape(1,200))

print(str(predicted_classes.reshape(1,200).int().tolist()).replace(',', '').replace(' ', ''))
