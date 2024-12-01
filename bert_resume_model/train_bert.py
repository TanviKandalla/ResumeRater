## Exploratory Data Analysis (EDA)
"""

#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

#Load the dataset
df = pd.read_csv("Preprocessed_Data.csv")

#Basic overview of the dataset
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

#Check for missing values and duplicates
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

#Category distribution
print("\nCategory Distribution:")
print(df['Category'].value_counts())

# Visualize the Category distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index)
plt.title('Distribution of Job Titles (Categories)')
plt.show()

# Analyze Common Words by Category
from nltk.corpus import stopwords

# Step 1: Set up stopwords
stop_words = set(stopwords.words('english'))

# Step 2: Group data by category and join all text for each category
category_texts = df.groupby('Category')['Text'].apply(lambda texts: ' '.join(texts)).reset_index()

# Step 3: Define a function to get the most common words for a category (excluding stopwords)
def get_top_words(text, n=10):
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    common_words = Counter(words).most_common(n)
    return common_words

# Step 4: Apply the function to each category and display top 10 words
for _, row in category_texts.iterrows():
    category = row['Category']
    text = row['Text']
    top_words = get_top_words(text, n=10)
    print(f"\nTop 10 Words for Category: {category}")
    for word, freq in top_words:
        print(f"{word}: {freq}")

"""##  Prepare Features for BERT training"""

#!pip install torch transformers scikit-learn

df.head()

df.isnull().sum()
df.info()

import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['Category'])
df.head(100)

# Split dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
train, temp = train_test_split(df, test_size=0.4, stratify=df['labels'])
val, test = train_test_split(temp, test_size=0.4, stratify=temp['labels'])
print(train.shape,val.shape,test.shape)

#!pip install datasets
#!pip install evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay

from transformers import BertTokenizer
from datasets import Dataset
import evaluate
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainerCallback

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to tokenize the text
def tokenize_function(examples):
    return tokenizer(examples['Text'], padding='max_length', truncation=True)

# Convert DataFrames to Hugging Face Datasets
train_df = Dataset.from_pandas(train)
val_df = Dataset.from_pandas(val)
test_df = Dataset.from_pandas(test)

# Tokenize datasets
train_df = train_df.map(tokenize_function, batched=True)
val_df= val_df.map(tokenize_function, batched=True)
test_df = test_df.map(tokenize_function, batched=True)

train_df.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_df.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_df.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

"""### BERT Model and Training"""

# Load BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df['labels'].unique()))

#call back to save full model
class CustomCallback(TrainerCallback):
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        torch.save(model,'train_bert_model.pt')

def compute_metrics(p):
    pred, labels = p
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred)
    pred = torch.argmax(pred, axis=1)
    acc = accuracy_score(labels, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='weighted')
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_df,
    eval_dataset=val_df,
    compute_metrics=compute_metrics,
    callbacks=[CustomCallback()]
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train
trainer.train()

trainer.evaluate(test_df)

import torch
from transformers import BertTokenizer, BertForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
model_path = 'train_bert_model.pt'
model = torch.load(model_path).to(device)  # Load the saved PyTorch model

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example text input for inference
text = "stephen greet cpa senior accountant 9 year experience establishing best accounting practices quickly growing tech companies ive learned value proractively identifying removing roadblocks allow accounting scale company want continue work early stage financial consumer startup work experience"

inputs = tokenizer(
    text,
    return_tensors="pt",  # Return PyTorch tensors
    padding=True,          # Pad to the maximum length
    truncation=True,       # Truncate to the max length if needed
    max_length=512         # Ensure max length is 512 tokens (default for BERT)
)

# Move input tensors to the same device as the model
inputs = {key: value.to(device) for key, value in inputs.items()}

# Perform inference
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class (output logits)
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=-1)

# Output the predicted class
print(f"Predicted Class: {predicted_class.item()}")
predicted_category = label_encoder.inverse_transform([predicted_class.item()])
print(f"Predicted Category: {predicted_category[0]}")

#######Example:
Predicted Class: 0
Predicted Category: Accountant
