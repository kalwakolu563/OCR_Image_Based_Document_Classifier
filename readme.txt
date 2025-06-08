python -m venv ocr_env
ocr_env\Scripts\activate
pip install -r requirements2.txt




python --version
C:\Users\Admin\AppData\Local\Programs\Python\Python311\python.exe -m venv ocr_env
ocr_env\Scripts\activate
python --version





Phase 2: Preprocessing with OpenCV
🔍 Steps:
Grayscale conversion

Noise removal with median blur

Adaptive thresholding

Morphological operations

Perspective correction (deskewing)

Resize to a standard dimension
Phase 3: OCR Implementation
🧠 Phase 4: Document Classification
✨ Approaches:
Text-Based (NLP):

Use OCR-extracted text.

Clean text (lowercase, remove stopwords).

Feature extraction: TF-IDF or BERT embeddings

Model: Logistic Regression / Random Forest / BERT → Fine-tune with transformers lib.

Image-Based (Vision):

Use CNN on document layout

Models: ResNet18, EfficientNet, or a custom CNN (PyTorch or TensorFlow)

Labels: Invoice, ID, Bank Statement, etc.

Phase 5: Information Extraction (Optional but Powerful)

Step-by-Step Architecture:
Image-Based Classification (CNN)

OCR (using Tesseract or EasyOCR)

Text-Based Classification (using BERT/DistilBERT)

Final Prediction (Ensemble / Fusion)