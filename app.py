from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, BertForSequenceClassification
from run import load_models, process_resume, find_similar_resumes
import pdfplumber
import torch

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BERT model and tokenizer
model_path = "bert_resume_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
bert_model = BertForSequenceClassification.from_pretrained(model_path)
bert_model.eval()

# Load similarity models
kmeans, cluster_results, cnn_model = load_models()

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

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

@app.post("/classify/")
async def classify_resume(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/jpeg", "image/png"]:
        return {"error": "Invalid file format. Only PDFs or images are accepted."}

    # Extract text for categorization
    pdf_text = extract_text_from_pdf(file.file)
    if not pdf_text:
        return {"error": "Failed to extract text from the uploaded PDF."}

    inputs = tokenizer(pdf_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    if predicted_class >= len(CATEGORIES):
        return {"error": f"Predicted class {predicted_class} is out of range."}
    category = CATEGORIES[predicted_class]

    # Reset file pointer and process the file for similarity
    file.file.seek(0)
    embedding = process_resume(file.file, cnn_model)
    similar_resumes = find_similar_resumes(embedding, kmeans, cluster_results)

    return {"index": predicted_class, "category": category, "similar_resumes": similar_resumes}

