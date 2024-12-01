Resume Scorer and Categorizer
=============================

This project implements two key components:

1.  **Category Classification**: Fine-tuning a BERT model for classifying resumes into predefined categories.
2.  **Resume Scoring**: Using CNN and VGG architectures to predict resume scores based on feature inputs.

* * * * *

Dataset Description
-------------------

The project uses the [Resume Atlas Dataset](https://huggingface.co/datasets/ahmedheakl/resume-atlas).\
For preprocessed datasets, refer to:

-   `Preprocessed_Data.csv`
-   `test_normalized_cv.csv`
-   `train_normalized_cv.csv`

* * * * *

Table of Contents
-----------------

1.  [Environment Setup](#environment-setup)
2.  [Dataset Preparation](#dataset-preparation)
3.  [Running the Code](#running-the-code)
    -   [Category Classification with BERT](#category-classification-with-bert)
    -   [Resume Scoring with CNN/VGG](#resume-scoring-with-cnnvgg)

* * * * *

1\. Environment Setup
---------------------

Ensure your system meets the following requirements:

-   **Python Version**: 3.8 or above
-   **Dependencies**:

    bash

    Copy code

    `pip install torch==1.11.0 transformers==4.28.1 scikit-learn==1.0.2 pandas==1.3.3 numpy==1.21.2 matplotlib==3.4.3 tqdm==4.62.3`

* * * * *

2\. Dataset Preparation
-----------------------

1.  Download the dataset:

    -   You can find the raw dataset here: [Resume Atlas Dataset](https://huggingface.co/datasets/ahmedheakl/resume-atlas).
    -   Preprocessed datasets:
        -   `Preprocessed_Data.csv`
        -   `test_normalized_cv.csv`
        -   `train_normalized_cv.csv`
2.  Ensure the dataset files are placed in the `data/` directory of your project.

* * * * *

3\. Running the Code
--------------------

### A. **Category Classification with BERT**

#### Training

1.  Train the BERT model for category classification:

    bash

    Copy code

    `python train_bert.py`

    -   This script fine-tunes a BERT model to classify resumes into predefined categories based on text features.
    -   **Input**: Text-based resumes.
    -   **Output**: A trained BERT model saved in the `bert_resume_model/` directory.

#### Predicting with the Backend API

1.  Use the trained BERT model to classify categories for new resumes via the FastAPI backend:

    bash

    Copy code

    `uvicorn app:app --reload --port 8000`

    -   API Endpoint: `http://127.0.0.1:8000/classify/`

* * * * *

### B. **Resume Scoring with CNN/VGG**

#### Training

1.  Train the shallow CNN model to predict resume scores:

    bash

    Copy code

    `python shallow_cnn.py`

    -   This script trains a CNN model for scoring resumes based on extracted features.
2.  The trained model will be saved in the `CNN Models/` directory.

#### Predicting with CLI

1.  Predict the score for a new resume:
    -   Use `run.py` to get scores and similar resumes:

        bash

        Copy code

        `python run.py`

    -   Follow the prompts to input the resume path.

* * * * *

### C. **Running the Frontend**

To serve the frontend for testing:

1.  Start a local HTTP server in the directory containing `index.html`:

    bash

    Copy code

    `python -m http.server 8080`

2.  Open your browser and navigate to:

    arduino

    Copy code

    `http://127.0.0.1:8080`

* * * * *

Additional Notes
----------------

-   **Frontend Hosting**:

    -   The frontend can be hosted on platforms like [Vercel](https://vercel.com/) or [Netlify](https://www.netlify.com/).
    -   Backend (FastAPI) can be hosted on platforms like [Render](https://render.com/).
-   **Folder Structure**:

    arduino

    Copy code

    `project/
    ├── data/
    │   ├── Preprocessed_Data.csv
    │   ├── test_normalized_cv.csv
    │   ├── train_normalized_cv.csv
    ├── bert_resume_model/
    ├── CNN Models/
    ├── app.py
    ├── train_bert.py
    ├── shallow_cnn.py
    ├── run.py
    ├── pad_data.py
    ├── index.html
    ├── script.js
    ├── style.css`

-   **Dependencies**:

    -   Ensure `torch`, `transformers`, and other libraries are properly installed before running the scripts.










