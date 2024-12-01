import torch
import torch.nn as nn
#import ResumeRater.pad_data as pad
import pad_data
import generate_embeddings
import os
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn as nn
import pandas as pd
#from ResumeRater import generate_embeddings
import pickle
from sklearn.metrics.pairwise import euclidean_distances


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



state_dict = torch.load("CNN Models\\shallow_cnn.pth")
model = shallow_CNN()
model.load_state_dict(state_dict)

intermediate_outputs = {}

def hook_fn(module, input, output):
    intermediate_outputs[module] = output

for name, layer in model.named_modules():
    if name in ["fc3"]:  
        layer.register_forward_hook(hook_fn)


#already_clustered = pd.read_csv("resume_clusters.csv")

#already_clustered = already_clustered["File Name"]
#already_clustered = np.array(already_clustered)

#print(already_clustered)
init_path = "C:\\Users\\Tanvi\\Desktop\\Resumes Datasets"
folders = os.listdir("C:\\Users\\Tanvi\\Desktop\\Resumes Datasets")
tensors = {}
count = 0

print("Processing images..")
for f in folders:
    #print(f)
    files = os.listdir(init_path + "\\" + f)
    for i in files:
        path = "C:\\Users\\Tanvi\\Desktop\\Resumes Datasets\\" + f + "\\" + i #replace with the path to data here
        subdir = os.listdir(path)
        for j in subdir:
            if count==5000:
                break
            image_path = path + "\\" + j
            #if image_path in already_clustered:
            #	continue
            if image_path in tensors.keys():
                continue
            image = Image.open(image_path)
            tensor = pad_data.resize_transform(image)
            # print(tensor.shape)
            tensors[image_path] = (tensor)
            count += 1
            # tensor = pad.pad_channels(tensor)
            # with torch.no_grad():
            #     output = model(input_tensor)
        if count==5000:
            break
    if count==5000:
        break
      

combined_tensor = (list(tensors.values()))
c = max([i.shape for i in combined_tensor])[0]
padded_combined = pad_data.pad_channels(combined_tensor, c = c)
padded_tensor = torch.stack(padded_combined)

print("Predicting outputs..")
with torch.no_grad():
    output = model(padded_tensor)

embeddings = intermediate_outputs[list(intermediate_outputs.keys())[0]]

objects = {}
for i in range(0, len(tensors.keys())):
    objects[list(tensors.keys())[i]] = embeddings[i]

class data_points():
    def __init__(self, file_name, embedding):
        self.file_name = file_name
        self.embedding = embedding


print("Creating dictionary..")
data_pts = []
for i in objects.keys():
    temp = data_points(i, objects[i])
    data_pts.append(temp)

embeddings = [pt.embedding for pt in data_pts]
embeddings = torch.stack(embeddings)

print("Clustering..")
kmeans = KMeans(n_clusters=24, random_state=42, max_iter = 300)
#kmeans = joblib.load("kmeans_model.pkl")
#kmeans.partial_fit(new_data)
kmeans.fit(embeddings.numpy())


with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Creating CSV with clusters
results = {"File Name": [pt.file_name for pt in data_pts], "Cluster": kmeans.labels_.tolist()}
results_df = pd.DataFrame(results)
results_df.to_csv("resume_clusters.csv", index=False, mode = 'a')
 
print("Cluster results saved to resume_clusters.csv!")

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

max_distances = []
for cluster_id in range(len(centroids)):
    # Get points belonging to the current cluster
    cluster_points = embeddings[labels == cluster_id]
    
    # Compute distances to the centroid
    distances = np.linalg.norm(cluster_points - centroids[cluster_id], axis=1)
    
    # Store the maximum distance
    max_distances.append(distances.max())


with open("centroid_distances.pkl", "wb") as f:
    pickle.dump(max_distances, f)
    
# Visualization using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings.numpy())

np.save("reduced_embeddings.npy", reduced_embeddings)
 
plt.figure(figsize=(12, 8))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_, cmap="tab10", s=50)
plt.colorbar(scatter, label="Cluster")
plt.title("Resumes Clustering Visualization (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)

for i, pt in enumerate(data_pts):
    plt.annotate(f"Resume {i}", (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.75)
 
plt.savefig("resume_clusters_visualization.png")
plt.show()

 



