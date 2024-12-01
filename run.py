import torch
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pad_data import resize_transform, pad_channels
import torch.nn as nn
import pickle


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



print("Loading model..")
# Load pre-trained components
kmeans = joblib.load("kmeans_model.pkl")
#embeddings = torch.load("embeddings.pth")
#reduced_embeddings = np.load("reduced_embeddings.npy")
cluster_results = pd.read_csv("resume_clusters.csv", header = 0)



print("Pre-processing image..")
# Process uploaded CV
uploaded_cv_path = input("Enter the path to your CV image file: ")
uploaded_image = Image.open(uploaded_cv_path)
uploaded_tensor = resize_transform(uploaded_image)
uploaded_tensor = pad_channels([uploaded_tensor], c=4)
uploaded_tensor = torch.stack(uploaded_tensor)


print("Finding embeddings..")
# Load model for embedding extraction
state_dict = torch.load("CNN Models/shallow_cnn.pth")
model = shallow_CNN()
model.load_state_dict(state_dict)
model.eval()

# extract embeddings
intermediate_outputs = {}
def hook_fn(module, input, output):
    intermediate_outputs[module] = output
model.fc3.register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model(uploaded_tensor)

uploaded_embedding = intermediate_outputs[list(intermediate_outputs.keys())[0]][0]


print("Figuring out which cluster it belongs to..")

print(uploaded_embedding.shape)
# Predict the cluster
predicted_cluster = kmeans.predict(uploaded_embedding.unsqueeze(0).numpy())[0]
print(f"The uploaded CV belongs to cluster: {predicted_cluster}")

with open("centroid_distances.pkl","rb") as f:
    cluster_radius = pickle.load(f)

predicted_centroid = kmeans.cluster_centers_[predicted_cluster]

distance = np.linalg.norm(uploaded_embedding.unsqueeze(0).numpy() - predicted_centroid, axis=1)

# print(distance)
# print(cluster_radius[predicted_cluster])
# print(distance / cluster_radius[predicted_cluster])

#print(cluster_radius,"______")

# Find similar resumes

# print(cluster_results.head())
similar_resumes = cluster_results[cluster_results["Cluster"] == predicted_cluster]["File Name"].sample(n = 10).values
print("\nTop 10 resumes from the same cluster:")
print(similar_resumes)
for resume in similar_resumes:
    print(resume)

import math
similarity_score = distance/cluster_radius[predicted_cluster]
similarity_score = math.ceil((1-similarity_score[0])*100)/10



print(similarity_score)

with open("Similar resumes.txt", "w") as f:
	f.write(str(similar_resumes) + "\nSimilarity score = " + str(similarity_score))

# Visualization
#plt.figure(figsize=(12, 8))
#plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap="tab10", s=50)
#plt.scatter(uploaded_embedding[0], uploaded_embedding[1], c="red", s=100, label="Uploaded CV")
#plt.colorbar(label="Cluster")
#plt.legend()
#plt.title("Resume Clusters (PCA)")
#plt.xlabel("PCA Component 1")
#plt.ylabel("PCA Component 2")
#plt.grid(True)
#plt.show()