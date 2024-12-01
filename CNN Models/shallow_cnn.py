import torch
import pandas as pd
import numpy as np
import warnings
import torch.nn as nn
from datetime import datetime

class shallow_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.block2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
        	nn.Dropout(0.6),
        	nn.Linear(13*13*128, 512),
        	nn.ReLU())

        self.fc2 = nn.Sequential(
        	nn.Dropout(0.6),
        	nn.Linear(512, 128),
        	nn.ReLU())

        self.fc3 = nn.Sequential(
        	nn.Dropout(0.6),
        	nn.Linear(128, 32),
        	nn.ReLU())  

        self.fc4= nn.Sequential(
        	nn.Linear(32, 3))


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

def main():
    X_train = torch.load("C:\\Users\\Tanvi\\Pictures\\OneDrive\\Pictures\\College\\NEU Semester 1\\Foundations of AI\\Final Project\\X_train_tensor.pt")
    X_test = torch.load("C:\\Users\Tanvi\\Pictures\\OneDrive\\Pictures\\College\\NEU Semester 1\\Foundations of AI\\Final Project\\X_test_tensor.pt")
    
    train_df = pd.read_csv("C:\\Users\\Tanvi\\Pictures\\OneDrive\\Pictures\\College\\NEU Semester 1\\Foundations of AI\\Final Project\\ResumeRater\\Preprocessed_data_CNN\\Train_normalized_CV.csv")
    train_df = train_df.drop("Unnamed: 0", axis = 1)
    
    for i in range(0, len(train_df)):
        for j in train_df.columns:
            if j == "File name":
                continue
            if (train_df[j][i]) < 0.5:
                train_df.loc[i, j] = 0.0
    
    test_df = pd.read_csv("C:\\Users\\Tanvi\\Pictures\\OneDrive\\Pictures\\College\\NEU Semester 1\\Foundations of AI\\Final Project\\ResumeRater\\Preprocessed_data_CNN\\Test_normalized_CV.csv")
    test_df = test_df.drop("Unnamed: 0", axis = 1)
    
    train_labels = train_df.drop(["File name", "Avg Format score"], axis = 1)
    y_train = train_labels.values
    y_train = torch.tensor(y_train)
    
    test_labels = test_df.drop(["File name", "Avg Format score"], axis = 1)
    y_test = test_labels.values
    y_test = torch.tensor(y_test)
    
    cnn = shallow_CNN()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.001, weight_decay = 1e-4)
    
    start_time = datetime.now()
    num_epochs = 10
    training_loss = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        # for i in range(0, y_train.shape[0]):
        inputs = X_train.float()
        inputs_noisy = inputs + 0.01 * torch.randn_like(inputs)
        labels = y_train.float()
        labels_noisy = labels + 0.01 * torch.randn_like(labels)
    
        #print(inputs.shape)
        # zero the parameter gradients
        optimizer.zero_grad()
    
        # forward + backward + optimize
        outputs = cnn(inputs_noisy)
        loss = criterion(outputs, labels_noisy)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"Epoch {epoch+1}:")
        end_time = datetime.now()
        print(f"Time taken to get here: {end_time-start_time}")
        print(f"Training loss: {running_loss}")
        training_loss.append(running_loss)
        # if epoch%50 == 0:
        #     torch.save(cnn.state_dict(),'shallow_cnn.pth')
            # with open("shallow_cnn.txt","w") as f:
            #     f.write(str(training_loss))
    
    print('Finished Training')
    
    torch.save(cnn.state_dict(), 'shallow_cnn.pth')
    
    mse_loss = nn.MSELoss()
    
    with torch.no_grad():
    	preds = cnn(X_test)
    loss = mse_loss(preds, y_test)
    
    print(preds)
    print("MSE Loss: ", loss.item())

if __name__ == "__main__":
    main()