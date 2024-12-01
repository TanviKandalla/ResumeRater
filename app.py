from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, BertForSequenceClassification
from run import load_models, process_resume, find_similar_resumes
import pdfplumber
import torch
import numpy as np
import pickle
import math

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BERT model and tokenizer for job categorization
model_path = "bert_resume_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertForSequenceClassification.from_pretrained(model_path)
bert_model.eval()

# Load similarity models for clustering and embeddings
kmeans, cluster_results, cnn_model = load_models()

# Load pre-computed cluster radius for similarity scoring
with open("centroid_distances.pkl", "rb") as f:
    cluster_radius = pickle.load(f)

# Define job categories
CATEGORIES = [
    "Education", "Mechanical Engineer", "Electrical Engineering", "Consultant", "Civil Engineer",
    "Management", "Human Resources", "Digital Media", "Accountant", "Java Developer",
    "Building and Construction", "Operations Manager", "Architecture", "Business Analyst",
    "Aviation", "Data Science", "Health and Fitness", "Arts", "Network Security Engineer",
    "DotNet Developer", "Apparel", "Banking", "Automobile", "Finance", "ETL Developer",
    "Agriculture", "Advocate", "DevOps", "Public Relations", "Designing", "Database",
    "BPO", "Sales", "Food and Beverages", "Testing", "SQL Developer", "Information Technology",
    "Web Designing", "Python Developer", "PMO", "SAP Developer", "Blockchain",
    "React Developer"
]

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

@app.post("/classify/")
async def classify_resume(file: UploadFile = File(...)):
    # Check if the file is a valid format
    if file.content_type not in ["application/pdf", "image/jpeg", "image/png"]:
        return {"error": "Invalid file format. Only PDFs or images are accepted."}

    try:
        # Extract text for job categorization
        pdf_text = extract_text_from_pdf(file.file)
        if not pdf_text:
            return {"error": "Failed to extract text from the uploaded PDF."}

        # Tokenize and predict job category using BERT
        inputs = tokenizer(pdf_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        # Validate predicted class
        if predicted_class >= len(CATEGORIES):
            return {"error": f"Predicted class {predicted_class} is out of range."}
        category = CATEGORIES[predicted_class]

        # Reset file pointer for similarity calculation
        file.file.seek(0)

        # Process the file for similarity using CNN and KMeans
        embedding = process_resume(file.file, cnn_model)

        # Predict the cluster
        predicted_cluster = kmeans.predict(embedding.unsqueeze(0).numpy())[0]

        # Compute similarity score
        predicted_centroid = kmeans.cluster_centers_[predicted_cluster]
        distance = np.linalg.norm(embedding.unsqueeze(0).numpy() - predicted_centroid, axis=1)
        similarity_score = math.ceil((1 - (distance / cluster_radius[predicted_cluster])[0]) * 100) / 10

        # Find similar resumes
        similar_resumes = cluster_results[cluster_results["Cluster"] == predicted_cluster]["File Name"].sample(
            n=10
        ).values.tolist()

        # Return results
        return {
            "index": predicted_class,
            "category": category,
            "similar_resumes": similar_resumes,
            "similarity_score": similarity_score,
        }

    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
