# GenAI & ML Fundamentals — Complete Guide

> **How to read this document:** This guide takes you from zero to a complete mental model of Generative AI and Machine Learning. Each concept builds on the ones before it. You'll get a clear definition, a real-world analogy, and working code for every topic.

## Road Map

```
PART 1 — THE BIG PICTURE          AI hierarchy, GenAI techniques (GANs, VAEs, Transformers, Diffusion), modalities
PART 2 — LANGUAGE & TEXT           NLP pipeline, text preprocessing, BoW, TF-IDF, subword tokenization
PART 3 — ML FUNDAMENTALS          Regression, classification, clustering, ML types, project lifecycle, ensembles
PART 4 — DATA & EVALUATION        Scaling, splits, CV, leakage, augmentation, metrics, overfitting, regularization
PART 5 — NEURAL NETWORKS          Perceptron → activations → FNN → CNNs → RNNs, optimizers, LR scheduling
PART 6 — REPRESENTATIONS          One-hot → embeddings → attention → positional encoding → Transformer architecture
PART 7 — TRANSFORMER MODELS       BERT, GPT, SBERT, dimensionality reduction, feature engineering
PART 8 — MODERN GenAI             LLMs, tokens, prompting, structured output, RAG, fine-tuning, agents, deployment, safety
```

---
---

# PART 1 — THE BIG PICTURE

---

# What is Gen AI?

GenAI (Generative AI) is a subset of AI that focuses on **creating new content or data**. Unlike traditional AI that only analyzes and interprets existing data, GenAI can generate entirely new text, images, music, code, and other forms of media.

**Analogy:** Think of traditional AI as a brilliant analyst who reads thousands of books and tells you the answers. GenAI is like that same analyst who can also *write* new books, compose music, or paint paintings based on everything they've learned.

## Where GenAI fits in the AI family tree

```
Artificial Intelligence (AI)
  └── Machine Learning (ML)           ← learns from data instead of explicit rules
        ├── Classical ML               ← decision trees, SVMs, random forests
        └── Deep Learning (DL)         ← neural networks with many layers
              ├── Discriminative        ← classifies or predicts (CNNs, BERT)
              └── Generative AI         ← creates new content (GPT, DALL-E, Stable Diffusion)
```

- **AI** = any system that mimics human intelligence (includes rule-based systems, search algorithms, etc.)
- **ML** = AI that learns patterns from data rather than being explicitly programmed
- **Deep Learning** = ML using multi-layer neural networks, excelling on unstructured data (images, text, audio)
- **GenAI** = Deep Learning models that generate new content rather than just classifying existing content

## Brief history — key milestones

| Year | Milestone | Why it matters |
|------|-----------|---------------|
| 1957 | Perceptron invented | First trainable neural network |
| 1986 | Backpropagation popularized | Made training multi-layer networks practical |
| 1997 | LSTM invented | Solved vanishing gradient for sequences |
| 2012 | AlexNet wins ImageNet | Deep learning revolution begins (CNNs) |
| 2013 | Word2Vec released | Words become vectors with meaning |
| 2014 | GANs introduced | Machines can generate realistic images |
| 2017 | "Attention Is All You Need" | The Transformer architecture — everything changes |
| 2018 | BERT released | Bidirectional pretraining redefines NLP benchmarks |
| 2018 | GPT-1 released | Decoder-only transformers show generative power |
| 2020 | GPT-3 (175B params) | Few-shot learning emerges at scale |
| 2022 | ChatGPT launched | LLMs reach mainstream adoption |
| 2022 | Stable Diffusion released | Open-source text-to-image diffusion model |
| 2023 | GPT-4, Claude 2, Llama 2 | Multimodal models, open-source race begins |
| 2024 | Claude 3.5, Llama 3, Gemini | 200K+ context windows, AI agents, tool use |

**Quick example — generating text with the Claude API:**

```python
import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a short poem about AI."}]
)
print(message.content[0].text)
```

---

# Core GenAI Techniques

## Generative Adversarial Networks (GANs)
Two neural networks compete against each other:
- **Generator** — creates fake data (the forger)
- **Discriminator** — tries to detect fake data (the detective)

They train together until the generator becomes so good that the discriminator can no longer tell real from fake.

**Analogy:** A counterfeiter printing fake banknotes vs. a bank inspector. The counterfeiter improves to fool the inspector; the inspector improves to catch the counterfeiter. Both get better through competition.

**Used for:** Generating realistic images, deepfakes, data augmentation.

## Variational Autoencoders (VAEs)
Compress data into a simplified "latent space" representation, then reconstruct (decode) it — with the ability to generate new variations.

**Analogy:** Like compressing an MP3 song. You lose some detail but keep the essence. VAEs add a twist: you can slightly modify the compressed version to generate new songs that sound similar but aren't identical.

**Used for:** Image generation, anomaly detection, data synthesis.

## Transformer Models
The dominant architecture today (GPT, BERT, Claude). Uses an "attention" mechanism to understand context across long sequences of text.

**Analogy:** Imagine reading a mystery novel. Transformers are like a reader who can simultaneously bookmark every important clue and understand how they all relate to each other — even if the clue is 50 pages away from the resolution.

**Used for:** LLMs (ChatGPT, Claude), translation, summarization, code generation.

## Diffusion Models
The technique behind modern image generation (Stable Diffusion, DALL-E 3, Midjourney). Works by learning to **reverse a noise-adding process** — start with pure noise and gradually denoise it into a coherent image.

**How it works:**
1. **Forward process (training):** Gradually add random noise to real images until they become pure static
2. **Reverse process (generation):** Train a neural network to undo the noise step by step — from static back to a crisp image

**Analogy:** Imagine a sculptor starting with a block of marble (noise) and chipping away at it, one tiny chip at a time, until a detailed statue emerges. The model learns exactly which "chips" to make at each step.

**Used for:** Text-to-image generation, image editing, inpainting, video generation.

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = pipe("a photo of an astronaut riding a horse on mars").images[0]
image.save("astronaut.png")
```

> **GANs vs Diffusion:** GANs are fast but hard to train (mode collapse). Diffusion models are slower but produce higher quality and more diverse outputs — that's why they dominate image generation today.

---

# GenAI Modalities

GenAI works across multiple types of content (modalities):

| Modality | What it does | Example tools |
|----------|-------------|---------------|
| **Text** | Generate, summarize, translate text | GPT-4, Claude, Gemini |
| **Vision** | Generate or understand images | DALL-E, Stable Diffusion, GPT-4V |
| **Audio** | Generate speech or music | ElevenLabs, Suno, Whisper |
| **Molecular** | Design molecules, drug discovery via genomic data | AlphaFold, ChemBERTa |

---

---
---

# PART 2 — LANGUAGE & TEXT

---

# What is NLP?

Natural Language Processing (NLP) is a branch of AI that enables machines to **understand, interpret, and generate human language** — essentially the technology behind every chatbot, search engine, voice assistant, and spell checker.

**Analogy:** NLP is like giving a computer the ability to read, listen, and write in plain English (or any language) — not just process binary commands. It bridges the gap between how humans communicate and how machines process information.

---

# NLP Enables You To:

- **Analyze and interpret text** — extract meaning, intent, and topics from written content
- **Interpret and contextualize spoken tokens** — understand emotional tone (sentimental analysis): "I love this product" = positive
- **Synthesize speech** — convert text to natural-sounding audio (voice assistants like Siri, Alexa talking back to you)

---

# Steps in NLP

## 1. Text Wrangling and Pre-processing

Raw text is messy. Before doing anything intelligent, you clean it up.

### Conversion
Normalize text: lowercase, remove accents, fix encoding.
```python
text = "Hello, WORLD! Héllo"
text = text.lower()  # "hello, world! héllo"
```

### Sanitization
Remove noise: HTML tags, special characters, extra whitespace.
```python
import re

text = "<p>Hello   world!!</p>"
text = re.sub(r'<.*?>', '', text)       # Remove HTML → "Hello   world!!"
text = re.sub(r'\s+', ' ', text).strip()  # Fix whitespace → "Hello world!!"
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation → "Hello world"
```

### Tokenization
Split text into individual units (tokens) — usually words or subwords.
```python
from nltk.tokenize import word_tokenize

text = "I love machine learning!"
tokens = word_tokenize(text)
# ['I', 'love', 'machine', 'learning', '!']
```

### Stemming
Chop word endings to their root form. Fast but rough — the result isn't always a real word.
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
print(stemmer.stem("running"))   # "run"
print(stemmer.stem("studies"))   # "studi"  ← rough, not a real word
print(stemmer.stem("happily"))   # "happili"
```

### Lemmatization
Map words to their true dictionary base form. Slower than stemming, but produces real words.
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos="v"))  # "run"
print(lemmatizer.lemmatize("better", pos="a"))   # "good"  ← smarter than stemming
print(lemmatizer.lemmatize("studies", pos="v"))  # "study"
```

> **Stemming vs Lemmatization:** Stemming is like hacking a word with a machete — fast but crude. Lemmatization is like looking it up in a dictionary — slower but accurate. Use lemmatization when output quality matters.

### Stop Words Removal
Remove common words that carry little meaning ("the", "is", "at", "and"). They add noise without adding information.
```python
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
tokens = ["I", "love", "machine", "learning", "and", "the", "amazing", "results"]
filtered = [w for w in tokens if w.lower() not in stop_words]
print(filtered)  # ['love', 'machine', 'learning', 'amazing', 'results']
```

> **When NOT to remove stop words:** Modern transformer models (BERT, GPT) actually need stop words — they carry grammatical meaning. Stop word removal is a classical NLP technique, not used with deep learning models.

---

## 2. Language Understanding (Structure / Syntax)

Once text is cleaned, we analyze its grammatical structure.

### Parts of Speech (POS) Tagging
Label each word with its grammatical role: noun, verb, adjective, etc.
```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a UK startup.")
for token in doc:
    print(token.text, "→", token.pos_)
# Apple → PROPN (proper noun)
# is → AUX (auxiliary verb)
# looking → VERB
# buying → VERB
# UK → PROPN
# startup → NOUN
```

### Chunking
Group POS tags into meaningful phrases (noun phrases, verb phrases).
```
"The quick brown fox" → [NP: The quick brown fox]
"jumped over the fence" → [VP: jumped] [PP: over the fence]
```
This is useful for extracting subject/object pairs from sentences.

### Dependency Parsing
Understand *how* words relate to each other — which word is the subject, object, modifier, etc.
```python
for token in doc:
    print(f"{token.text:10} --({token.dep_:10})--> {token.head.text}")
# Apple      --(nsubj     )--> looking
# is         --(aux       )--> looking
# looking    --(ROOT      )--> looking
# buying     --(pcomp     )--> at
```

### Constituency Parsing
Break a sentence into a nested hierarchical tree of phrases. More structural, used in formal NLP tasks.
```
(S
  (NP Apple)
  (VP is
    (VP looking
      (PP at
        (VP buying
          (NP a UK startup))))))
```

---

## 3. Processing and Functionality

Now use what we understood to actually do something useful.

### Named Entity Recognition (NER)
Automatically identify real-world entities in text: people, places, organizations, dates, amounts.
```python
doc = nlp("Elon Musk founded SpaceX in 2002 in Hawthorne, California.")
for ent in doc.ents:
    print(ent.text, "→", ent.label_)
# Elon Musk → PERSON
# SpaceX → ORG
# 2002 → DATE
# Hawthorne → GPE (geo-political entity)
# California → GPE
```

### N-gram Identification
Sequences of N consecutive words. Used for keyword extraction and language modeling.
```python
from nltk.util import ngrams

tokens = ["I", "love", "machine", "learning", "deeply"]
bigrams  = list(ngrams(tokens, 2))  # [('I','love'), ('love','machine'), ...]
trigrams = list(ngrams(tokens, 3))  # [('I','love','machine'), ...]
```

### Sentiment Analysis
Classify text as positive, negative, or neutral — understand emotional tone.
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
results = classifier([
    "I absolutely loved the movie!",
    "The service was terrible and rude.",
    "It was okay, nothing special."
])
for r in results:
    print(r)
# {'label': 'POSITIVE', 'score': 0.9998}
# {'label': 'NEGATIVE', 'score': 0.9995}
# {'label': 'NEGATIVE', 'score': 0.6329}
```

### Information Extraction
Pull structured data from unstructured text — e.g., extract dates, prices, and parties from a contract automatically.

### Information Retrieval
Find relevant documents from a large corpus given a query — the core of how search engines work. Modern IR uses embeddings + vector search (see SBERT below).

### Question Answering
Given a context passage, answer a specific question by finding the answer span within it.
```python
from transformers import pipeline

qa = pipeline("question-answering")
result = qa(
    question="Where was Einstein born?",
    context="Albert Einstein was born in Ulm, in the Kingdom of Württemberg in Germany, in 1879."
)
print(result['answer'])  # "Ulm"
print(result['score'])   # confidence score
```

### Topic Modeling
Automatically discover abstract topics across a collection of documents — no labels needed.
```python
# Using gensim LDA (Latent Dirichlet Allocation)
# Each document = mix of topics; each topic = mix of words
# Example output after training on news articles:
# Topic 0 → ["bank", "money", "loan", "finance", "interest"]   → Finance
# Topic 1 → ["river", "water", "fish", "dam", "flood"]         → Environment
# Topic 2 → ["election", "vote", "candidate", "party", "poll"] → Politics
```

---

# Text Representation: Bag-of-Words and TF-IDF

Before embeddings (Word2Vec, BERT), text had to be converted to numbers using **counting-based methods**. These are still used in production for lightweight NLP tasks.

## Bag-of-Words (BoW)
Represent each document as a **vector of word counts**. Ignores word order entirely — just "what words are present and how often."

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "I love machine learning",
    "machine learning is amazing",
    "I love deep learning"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
# ['amazing', 'deep', 'is', 'learning', 'love', 'machine']
print(X.toarray())
# [[0, 0, 0, 1, 1, 1],   ← "I love machine learning"
#  [1, 0, 1, 1, 0, 1],   ← "machine learning is amazing"
#  [0, 1, 0, 1, 1, 0]]   ← "I love deep learning"
```

**Limitation:** "I love dogs" and "Dogs love I" produce identical vectors — word order is lost.

## TF-IDF (Term Frequency — Inverse Document Frequency)
An improvement over BoW: **words that appear in many documents get downweighted** (common words like "the" become less important), while rare but meaningful words get boosted.

- **TF** = how often a word appears in THIS document (higher = more relevant to this doc)
- **IDF** = how rare a word is across ALL documents (rarer = more informative)
- **TF-IDF = TF × IDF** — a word scores high if it's frequent in this document but rare overall

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
print(X.toarray().round(2))
# "learning" appears in all 3 docs → low IDF → lower weight
# "amazing" appears in only 1 doc → high IDF → higher weight
```

**Real-world use:** TF-IDF is still used in production for:
- Search engine ranking (the original PageRank used TF-IDF)
- Spam detection
- Document similarity (faster and cheaper than embeddings for simple cases)
- Feature extraction before feeding into classical ML models

> **BoW/TF-IDF vs Embeddings:** BoW and TF-IDF are sparse, high-dimensional, and ignore word meaning ("happy" and "joyful" are as different as "happy" and "car"). Embeddings are dense, low-dimensional, and capture semantic meaning. Use TF-IDF when you need speed and simplicity. Use embeddings when you need understanding.

---

# LLM Tokenization (BPE, WordPiece, SentencePiece)

The NLP tokenization above splits text into **whole words**. Modern LLMs use **subword tokenization** — splitting words into smaller meaningful pieces. This is a fundamentally different approach.

**Why subwords?** Word-level tokenization has two fatal problems:
1. **Vocabulary explosion** — millions of unique words across all languages
2. **Unknown words** — any word not in the vocabulary becomes `[UNK]`

Subword tokenization solves both: build a manageable vocabulary (~30K-100K pieces) that can represent ANY word by combining pieces.

**Analogy:** Instead of having a dictionary entry for every word ever written, you have a Lego set of common word-parts. "unhappiness" becomes `["un", "happiness"]` or `["un", "happ", "iness"]`. You can build any word from the pieces.

## Common Subword Algorithms

| Algorithm | Used by | How it works |
|-----------|---------|-------------|
| **BPE** (Byte-Pair Encoding) | GPT, Llama, Claude | Iteratively merge most frequent character pairs |
| **WordPiece** | BERT | Similar to BPE but uses likelihood maximization |
| **SentencePiece** | T5, XLM-RoBERTa | Language-agnostic, works on raw text (no pre-tokenization) |

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love transformers and unbelievable tokenization!"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['i', 'love', 'transformers', 'and', 'un', '##believe', '##able', 'token', '##ization', '!']
# Notice: "unbelievable" → ["un", "##believe", "##able"] (subword pieces)

# Token IDs (what the model actually sees)
ids = tokenizer.encode(text)
print(ids)
# [101, 1045, 2293, 19081, 1998, 4895, 16429, 3085, 19204, 3989, 999, 102]

print(f"Text has {len(text.split())} words but {len(tokens)} tokens")
```

> **Rule of thumb:** In English, 1 token ≈ 0.75 words (or ~4 characters). "ChatGPT is great" = 4 words but 5 tokens.

---

---
---

# PART 3 — ML FUNDAMENTALS

---

# Regression

Regression finds a mathematical function that maps inputs to a **continuous numerical output**.

**Analogy:** Predicting tomorrow's temperature. You look at past data (humidity, season, historical averages) and draw the best-fit line through all the data points. Any new day's data maps to a point on (or near) that line giving you the predicted temperature.

**Error metrics — how wrong is the model?**
- **MAE** (Mean Absolute Error): average of absolute differences — intuitive to interpret ("off by $5k on average")
- **MSE** (Mean Squared Error): penalizes large errors more heavily — sensitive to outliers
- **RMSE** (Root MSE): square root of MSE, same units as the target — most commonly reported
- **R² (R-squared)**: proportion of variance explained by the model. R²=1 means perfect fit, R²=0 means the model is no better than predicting the mean. Negative R² means worse than the mean. This is the metric you'll see most in regression reports

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# House price prediction: predict price from square footage
X = [[1000], [1500], [2000], [2500], [3000]]  # sq ft
y = [200000, 280000, 350000, 420000, 500000]  # price ($)

model = LinearRegression()
model.fit(X, y)

# Predict price for a 1800 sq ft house
prediction = model.predict([[1800]])
print(f"Predicted price: ${prediction[0]:,.0f}")  # ~$316,000

# Evaluate on training data
y_pred = model.predict(X)
print(f"MAE:  ${mean_absolute_error(y, y_pred):,.0f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y, y_pred)):,.0f}")
```

**Different Regression algorithms:**
- **Linear Regression** — fits a straight line (fast, interpretable)
- **Polynomial Regression** — fits curves for non-linear data
- **Ridge / Lasso** — linear regression with regularization to prevent overfitting
- **Random Forest Regressor / XGBoost** — powerful, handles non-linearity automatically

---

# Classification

Classification assigns input data to one of several **discrete categories (classes)**.

**Analogy:** A spam filter. Every email is either "spam" or "not spam." The model learns from thousands of labeled examples, then classifies any new email it's never seen before.

**Common algorithms:**
- **Logistic Regression** — simple, fast, great baseline despite the name
- **Decision Tree / Random Forest** — interpretable, handles non-linear data well
- **Neural Networks** — most powerful for complex patterns (images, text)
- **Naive Bayes** — very fast, excellent for text classification
- **K-Nearest Neighbors (KNN)** — classifies based on similarity to nearby examples
- **Support Vector Machines (SVM)** — finds the optimal decision boundary between classes

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Classify iris flowers into 3 species based on petal/sepal measurements
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

> **Regression vs Classification:** Regression predicts a number (price, temperature). Classification predicts a category (spam/not spam, cat/dog/bird).

---

# Clustering

Clustering groups **unlabeled data** into clusters based on similarity — the model discovers the categories on its own, without being told what they are.

**Analogy:** Imagine dumping 1,000 news articles on your desk with no labels. Clustering automatically groups them: politics together, sports together, tech together — even though nobody defined those categories in advance.

**Common algorithms:**
- **K-Means** — partition into K groups by distance to centroids (most popular, fast)
- **K-Medoids** — like K-Means but uses actual data points as centers (more robust to outliers)
- **DBSCAN** (Density-based) — finds clusters of arbitrary shape, handles noise and outliers well
- **Hierarchical** — builds a tree of clusters (dendrogram), no need to specify K upfront

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate 300 data points naturally grouped into 3 blobs
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Cluster into 3 groups (without using the true labels)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200, c='red', marker='X', label='Centroids'
)
plt.title("K-Means Clustering")
plt.legend()
plt.show()
```

> **Key difference from Classification:** Classification needs labeled data to learn from. Clustering discovers patterns entirely on its own from unlabeled data.

---

# Types of Machine Learning

| Type | Data needed | How it learns | Example use case |
|------|-------------|---------------|-----------------|
| **Supervised** | Labeled (input + correct output) | Learns from input→output pairs | Spam detection, price prediction |
| **Unsupervised** | No labels | Finds hidden patterns alone | Customer segmentation, anomaly detection |
| **Reinforcement** | Environment + reward signal | Trial and error, maximize reward | Game-playing AI (AlphaGo), robotics |
| **Semi-supervised** | Mostly unlabeled, few labeled | Leverages unlabeled structure + a few labels | Medical imaging (few annotated scans) |
| **Self-supervised** | Unlabeled (creates own labels) | Predicts part of input from other parts | BERT, GPT pretraining |
| **Multi-instance** | Bags of instances, bag-level labels | Label assigned to groups, not individuals | Drug activity prediction |
| **Inductive** | Training examples | Generalizes rules to unseen data | Most standard ML workflows |
| **Deductive** | General rules | Applies rules to specific new cases | Expert/rule-based systems |
| **Transductive** | Training + specific test set | Predicts only for that specific unlabeled set | Label propagation on graphs |
| **Multi-task** | Multiple tasks' labels | Learns several tasks simultaneously | NLP model doing NER + POS + parsing at once |
| **Active** | Small labeled pool + oracle | Queries human for labels on uncertain examples | When labeling is expensive |
| **Online** | Continuous data stream | Updates model continuously as new data arrives | Fraud detection, stock prediction |
| **Transfer** | Pre-trained model + new task data | Reuses old knowledge for a new task | Fine-tuning BERT on your domain |
| **Ensemble** | Multiple trained models | Combines predictions of many models | Random Forest, XGBoost, model stacking |

---

# The ML Project Lifecycle

In real-world projects, training a model is only ~20% of the work. Here's the full lifecycle:

```
1. Define Problem → 2. Collect Data → 3. Clean & Explore → 4. Feature Engineering
                                                                     ↓
8. Monitor & Retrain ← 7. Deploy ← 6. Evaluate ← 5. Train & Tune Model
```

| Phase | What happens | Tools commonly used |
|-------|-------------|-------------------|
| **1. Problem definition** | Define the business goal, success metric, and what "good enough" looks like | Stakeholder meetings, domain experts |
| **2. Data collection** | Gather raw data from databases, APIs, web scraping, labeling services | SQL, pandas, Label Studio, Scale AI |
| **3. Data cleaning & EDA** | Handle missing values, outliers, duplicates. Explore distributions and correlations | pandas, matplotlib, seaborn |
| **4. Feature engineering** | Create, transform, and select features that help the model learn | pandas, sklearn transformers |
| **5. Model training & tuning** | Train multiple models, tune hyperparameters, compare performance | sklearn, XGBoost, PyTorch, Optuna |
| **6. Evaluation** | Evaluate on held-out test set using the right metrics for your problem | sklearn.metrics, cross-validation |
| **7. Deployment** | Serve the model in production via API, batch pipeline, or edge device | FastAPI, Docker, AWS SageMaker, MLflow |
| **8. Monitoring & retraining** | Track model drift (when real-world data shifts away from training data), retrain periodically | Evidently AI, Weights & Biases, MLflow |

**Experiment tracking in practice:** In real projects, you train dozens of model variants. Tools like **MLflow** and **Weights & Biases** log every experiment (hyperparameters, metrics, artifacts) so you can compare and reproduce results.

```python
import mlflow

mlflow.set_experiment("house-price-prediction")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("rmse", 15420.5)
    mlflow.log_metric("r2", 0.94)
    mlflow.sklearn.log_model(model, "model")
    # All params, metrics, and the model itself are now versioned and reproducible
```

> **The 80/20 rule of ML:** 80% of the work is data preparation, cleaning, and feature engineering. Only 20% is actual model training. The fanciest algorithm won't save you from bad data.

---

# Classical ML

Classical ML refers to algorithms that work well **without deep neural networks** — usually faster to train, more interpretable, and often better for structured/tabular data.

**When to use Classical ML over Deep Learning:**
- Your dataset is small (< 100K rows typically)
- You need interpretability (medical, legal, financial decisions)
- You have structured/tabular data (spreadsheet-style)
- You want fast training, low compute cost

**Most common classical algorithms and when to reach for each:**

| Algorithm | Best for |
|-----------|---------|
| Linear / Logistic Regression | Best first baseline, highly interpretable |
| Decision Tree | When you need to explain decisions step-by-step |
| Random Forest | General-purpose tabular data workhorse |
| XGBoost / LightGBM | Best performance on tabular data competitions |
| SVM | High-dimensional data, image classification (pre-deep-learning) |
| Naive Bayes | Very fast text classification |
| KNN | Small datasets, no training needed, simple logic |

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

data = load_breast_cancer()

# A clean sklearn pipeline: scale features, then classify
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5-fold cross-validation — more reliable than a single train/test split
scores = cross_val_score(pipeline, data.data, data.target, cv=5, scoring='accuracy')
print(f"5-fold CV accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
# Typically: ~96-97% on breast cancer dataset
```

## Ensemble Methods — Why They Win Competitions

Ensemble methods **combine multiple models** to produce better predictions than any single model alone. They dominate Kaggle competitions and are heavily used in production.

### Bagging (Bootstrap Aggregating)
Train multiple models on **random subsets** of the data, then average their predictions. Reduces variance (overfitting).
- **Random Forest** = bagging with decision trees + random feature subsets per split

### Boosting
Train models **sequentially** — each new model focuses on the errors the previous models got wrong. Reduces bias (underfitting).
- **XGBoost** / **LightGBM** / **CatBoost** = the workhorses of tabular ML in production

### Stacking
Train multiple different model types, then train a **meta-model** on top that learns how to best combine their predictions.

```python
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Simple ensemble: combine 3 different models via majority vote
ensemble = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=10000)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('gb', GradientBoostingClassifier(n_estimators=100))
], voting='soft')  # 'soft' = average probabilities, 'hard' = majority vote

scores = cross_val_score(ensemble, data.data, data.target, cv=5)
print(f"Ensemble accuracy: {scores.mean():.2%}")
# Often beats any individual model
```

> **In production:** XGBoost and LightGBM are the most commonly deployed ML models for tabular data — banks (fraud detection), e-commerce (recommendation scoring), healthcare (risk prediction). They're fast, accurate, and handle messy real-world data well.

---

---
---

# PART 4 — DATA & EVALUATION

---

# Feature Scaling

Feature scaling transforms your features so they're on a **comparable scale**. Many algorithms (SVM, KNN, neural networks, gradient descent) are sensitive to feature magnitudes — without scaling, a feature ranging 0-1000 will dominate one ranging 0-1.

**Analogy:** Imagine comparing height (in cm, ~150-200) and weight (in kg, ~50-100) directly. Without scaling, height numbers are always bigger and would dominate any distance-based calculation, even though both features matter equally.

## Two Common Approaches

| Method | Formula | Output range | When to use |
|--------|---------|-------------|-------------|
| **StandardScaler** | (x - mean) / std | Centered at 0, ~[-3, 3] | Default choice. Most ML algorithms |
| **MinMaxScaler** | (x - min) / (max - min) | [0, 1] | When you need bounded values (e.g., neural nets) |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

data = np.array([[100], [500], [1000], [5000], [10000]])

standard = StandardScaler().fit_transform(data)
minmax = MinMaxScaler().fit_transform(data)

print("Original:  ", data.flatten())         # [100, 500, 1000, 5000, 10000]
print("Standard:  ", standard.flatten().round(2))  # [-0.86, -0.73, -0.57, 0.72, 1.44]
print("MinMax:    ", minmax.flatten().round(2))     # [0.0, 0.04, 0.09, 0.49, 1.0]
```

> **When scaling is NOT needed:** Tree-based models (Decision Tree, Random Forest, XGBoost) are immune to feature scaling because they split on thresholds, not distances.

---

# Train / Validation / Test Splits

You need to split your data into **three** separate sets — not two — to build a reliable model.

| Set | Purpose | Typical % |
|-----|---------|-----------|
| **Training set** | Model learns from this | 70% |
| **Validation set** | Tune hyperparameters and check for overfitting during training | 15% |
| **Test set** | Final, one-time evaluation of real-world performance | 15% |

**Analogy:** Studying for an exam. The training set is your textbook (you learn from it). The validation set is practice tests (you use them to adjust your study strategy). The test set is the real exam (you take it once, and it reveals your true knowledge).

**Why not just train/test?** If you tune hyperparameters using the test set, you're indirectly training on it — your test score will be overoptimistic and won't reflect real-world performance.

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# First split: separate out the test set (never touch until final evaluation)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Second split: separate train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
)
# 0.18 of 0.85 ≈ 0.15 of total → roughly 70/15/15

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

> **Golden rule:** The test set is sacred. Touch it **once** — at the very end — to report your final metric.

---

# Cross-Validation

Cross-validation gives you a **more reliable estimate** of model performance than a single train/test split — especially important when your dataset is small.

**How K-Fold Cross-Validation works:**
1. Split the data into K equal parts ("folds")
2. For each fold: use it as the validation set, train on the remaining K-1 folds
3. Average the K validation scores → your final performance estimate

**Analogy:** Instead of taking one practice exam, you take 5 different practice exams, each covering a different portion of the material. Your average score across all 5 is a much more reliable predictor of your real exam performance.

```
5-Fold CV:
Fold 1: [VAL][Train][Train][Train][Train] → score₁
Fold 2: [Train][VAL][Train][Train][Train] → score₂
Fold 3: [Train][Train][VAL][Train][Train] → score₃
Fold 4: [Train][Train][Train][VAL][Train] → score₄
Fold 5: [Train][Train][Train][Train][VAL] → score₅
Final score = mean(score₁ ... score₅)
```

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# StratifiedKFold ensures each fold has the same class distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

print(f"Fold scores: {scores}")
print(f"Mean accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
# Much more trustworthy than a single split!
```

> **When to use:** Always use CV for model selection and hyperparameter tuning. Only skip it when your dataset is very large (>100K samples) and a single split is representative enough.

---

# Data Leakage

Data leakage is when **information from outside the training set "leaks" into the model during training**, giving unrealistically good performance that won't hold in production.

**Analogy:** Taking an exam where you accidentally saw the answer key beforehand. Your score is 100%, but you didn't actually learn anything — and you'll fail any real test.

**The most common mistake — scaling before splitting:**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# ❌ WRONG: fit scaler on ALL data, then split
# The scaler "sees" test data statistics during training → leakage
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # learns mean/std from ENTIRE dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# ✅ RIGHT: split first, fit scaler on train only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # learn from train only
X_test = scaler.transform(X_test)          # apply same transform, no fitting
```

**Other common sources of leakage:**
- Using future data to predict the past (time series)
- Including the target variable (or a proxy for it) as a feature
- Data augmentation before splitting (same augmented image in train and test)

> **How to spot leakage:** If your model performs suspiciously well (99%+ accuracy on a hard problem), check for leakage before celebrating.

### Special case: Time Series Splits
For time-ordered data, **you can't shuffle randomly** — future data would leak into training. Always split chronologically:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# Fold 1: train=[Jan-Mar], test=[Apr]
# Fold 2: train=[Jan-Apr], test=[May]
# Fold 3: train=[Jan-May], test=[Jun]  ← training set grows each fold
```

---

# Data Augmentation

Data augmentation **creates new training examples by applying transformations** to existing data — effectively increasing your dataset size without collecting new data. Critical when labeled data is scarce.

**For images:**
```python
from torchvision import transforms

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(224, padding=20),
])
# One cat photo → dozens of slightly different cat photos
# The model learns "cat" regardless of orientation, lighting, or position
```

**For text (common techniques):**
- **Synonym replacement:** "The movie was great" → "The film was excellent"
- **Back-translation:** English → French → English (paraphrases automatically)
- **Random insertion/deletion:** Add or remove words while preserving meaning

**In production:** Data augmentation is standard in virtually every computer vision pipeline. ImageNet models use extensive augmentation during training. For NLP, augmentation is less common with large LLMs since they're pretrained on massive corpora, but it's still used for fine-tuning smaller models on limited labeled data.

---

# Evaluation Metrics

Accuracy alone is often **misleading**. Different metrics answer different questions about your model.

## Confusion Matrix

The foundation of all classification metrics:

```
                  Predicted
                 Pos    Neg
Actual  Pos  [  TP  |  FN  ]    ← TP: correctly predicted positive
        Neg  [  FP  |  TN  ]    ← FP: falsely predicted positive (Type I error)
                                   FN: missed a positive (Type II error)
```

## Key Metrics

| Metric | Formula | Question it answers | Prioritize when... |
|--------|---------|--------------------|--------------------|
| **Accuracy** | (TP+TN) / total | Overall, how often are we right? | Classes are balanced |
| **Precision** | TP / (TP+FP) | Of predicted positives, how many are correct? | False positives are costly (spam filter) |
| **Recall** | TP / (TP+FN) | Of actual positives, how many did we catch? | False negatives are costly (cancer detection) |
| **F1 Score** | 2 × (P×R)/(P+R) | Balanced measure of precision and recall | You need to balance both |
| **AUC-ROC** | Area under ROC curve | How well can the model separate classes at all thresholds? | You want a single threshold-independent score |

**Analogy:**
- **Precision** = "When the fire alarm goes off, is there actually a fire?" (low FP)
- **Recall** = "When there is a fire, does the alarm go off?" (low FN)

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
```

> **Rule of thumb:** Use **accuracy** only when classes are balanced. For imbalanced data, always report **precision**, **recall**, **F1**, and **AUC-ROC**.

---

# Imbalanced Datasets

When one class heavily outnumbers another (e.g., 99% not-fraud, 1% fraud), standard accuracy becomes **meaningless** — a model that always predicts "not fraud" scores 99% accuracy while catching zero fraud cases.

**Analogy:** A doctor who tells every patient "you're healthy" would be right 99% of the time, but that 1% they missed could be life-threatening.

**Solutions:**

| Strategy | How it works |
|----------|-------------|
| **Class weights** | Tell the model to penalize errors on the minority class more |
| **Oversampling (SMOTE)** | Generate synthetic minority samples to balance the dataset |
| **Undersampling** | Remove majority samples (loses data, use when dataset is large) |
| **Better metrics** | Use F1, AUC-ROC, precision/recall instead of accuracy |

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

# Create an imbalanced dataset: 95% class 0, 5% class 1
X, y = make_classification(n_samples=2000, weights=[0.95, 0.05], random_state=42)

# ❌ Without handling imbalance
clf_default = RandomForestClassifier(random_state=42)
clf_default.fit(X, y)

# ✅ With class_weight='balanced' — penalizes minority misses more
clf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42)
clf_balanced.fit(X, y)

print("Default:")
print(classification_report(y, clf_default.predict(X), digits=3))
print("Balanced:")
print(classification_report(y, clf_balanced.predict(X), digits=3))
```

---

# Overfitting and Underfitting

Two fundamental failure modes of any ML model:

**Underfitting** — the model is **too simple** to capture patterns in the data. High error on both training and test data. Like a student who barely studied.

**Overfitting** — the model **memorized the training data** including its noise and quirks. Low error on training data, high error on test data. Like a student who memorized exam answers verbatim but can't answer rephrased questions.

**Analogy:** Fitting a curve through data points.
- **Underfit:** Drawing a straight line through clearly curved data — too simplistic
- **Good fit:** Drawing a smooth curve that follows the general trend
- **Overfit:** Drawing a wild, zigzagging line that passes through every single point — including noise

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

np.random.seed(42)
X = np.sort(np.random.rand(30, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(30) * 0.3  # sine + noise

X_train, X_test = X[:20], X[20:]
y_train, y_test = y[:20], y[20:]

for degree in [1, 4, 15]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    train_err = mean_squared_error(y_train, model.predict(X_train))
    test_err = mean_squared_error(y_test, model.predict(X_test))
    print(f"Degree {degree:2d}: Train MSE={train_err:.4f}, Test MSE={test_err:.4f}")

# Degree  1: Train MSE=0.2190, Test MSE=0.3847  ← underfit (too simple)
# Degree  4: Train MSE=0.0737, Test MSE=0.1095  ← good fit
# Degree 15: Train MSE=0.0132, Test MSE=9.8753  ← overfit (memorized noise!)
```

**How to fix each:**

| Problem | Signs | Fixes |
|---------|-------|-------|
| Underfitting | High train error, high test error | More complex model, more features, less regularization |
| Overfitting | Low train error, high test error | More training data, regularization, simpler model, dropout, early stopping |

### Early Stopping — The Most Practical Overfitting Prevention

Monitor validation loss during training. **Stop training when validation loss starts increasing** even though training loss continues to decrease — that's the exact moment overfitting begins.

```python
# PyTorch early stopping pattern (used in virtually every deep learning project)
best_val_loss = float('inf')
patience = 5           # how many epochs to wait after val loss stops improving
patience_counter = 0

for epoch in range(1000):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")  # save the best checkpoint
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load the best model (not the last one!)
model.load_state_dict(torch.load("best_model.pt"))
```

> **In production:** Early stopping is used in almost every neural network training pipeline. Libraries like PyTorch Lightning and Keras have built-in `EarlyStopping` callbacks.

---

# Bias-Variance Tradeoff

Every model's error comes from two sources:

- **Bias** — error from wrong assumptions. A too-simple model makes systematic errors. *High bias = underfitting.*
- **Variance** — error from sensitivity to training data noise. A too-complex model changes wildly with different training samples. *High variance = overfitting.*

**You can't minimize both simultaneously** — reducing one typically increases the other. The goal is the sweet spot.

**Analogy:** Throwing darts at a target:
- **High bias, low variance:** All darts land in the same wrong spot (consistent but inaccurate — like always predicting "average")
- **Low bias, high variance:** Darts scatter everywhere but centered on the bullseye (accurate on average but inconsistent)
- **Low bias, low variance:** Darts cluster tightly on the bullseye (the goal)

| | Low Variance | High Variance |
|---|---|---|
| **Low Bias** | Ideal model | Overfitting (complex model, little data) |
| **High Bias** | Underfitting (too simple) | Worst case (wrong assumptions + noisy) |

> **Practical takeaway:** Start with a simple model (high bias, low variance). Gradually increase complexity until test error stops improving. When test error starts rising while training error keeps falling — you've crossed into overfitting territory.

---

# Regularization

Regularization **penalizes model complexity** to prevent overfitting. It adds a cost for having large weights, encouraging the model to find simpler solutions.

**Analogy:** A teacher grading essays who deducts points for unnecessary complexity. "Get to the point — don't use 50 words when 10 will do." This forces more concise, generalizable answers.

## L1 (Lasso) and L2 (Ridge)

| Type | Penalty | Effect | Best for |
|------|---------|--------|----------|
| **L1 (Lasso)** | Sum of absolute weights | Drives some weights to exactly zero → automatic feature selection | Sparse models, identifying important features |
| **L2 (Ridge)** | Sum of squared weights | Shrinks all weights toward zero (but none to exactly zero) | When all features matter but you want smaller weights |
| **ElasticNet** | L1 + L2 combined | Best of both worlds | When you want feature selection + shrinkage |

```python
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

np.random.seed(42)
X = np.sort(np.random.rand(30, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.randn(30) * 0.3

X_train, X_test = X[:20], X[20:]
y_train, y_test = y[:20], y[20:]

# High-degree polynomial WITHOUT regularization → overfits
plain = make_pipeline(PolynomialFeatures(15), LinearRegression())
plain.fit(X_train, y_train)

# Same polynomial WITH Ridge regularization → tamed
ridge = make_pipeline(PolynomialFeatures(15), Ridge(alpha=1.0))
ridge.fit(X_train, y_train)

print(f"No regularization: Test MSE = {mean_squared_error(y_test, plain.predict(X_test)):.4f}")
print(f"Ridge (L2):        Test MSE = {mean_squared_error(y_test, ridge.predict(X_test)):.4f}")
# Ridge dramatically reduces test error by preventing wild coefficient values
```

**Neural network regularization:** In deep learning, **dropout** serves the same purpose — randomly disabling neurons during training forces the network to not rely on any single pathway. (See Dropout section below.)

---

---
---

# PART 5 — NEURAL NETWORKS

---

# Perceptron

The perceptron is the simplest building block of neural networks — a **single artificial neuron** capable of binary classification.

**Analogy:** Imagine a light switch controlled by multiple dials (inputs). Each dial has a weight (how much it matters). If the weighted combination of all dials exceeds a threshold, the light turns ON (output = 1). Otherwise it stays OFF (output = 0).

## How it works:

1. Each input `x` is multiplied by its weight `w`
2. Sum all weighted inputs plus a bias: `z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`
3. Apply a step activation function: output = 1 if `z ≥ 0`, else 0
4. During training: if wrong, adjust weights in the direction of the error

```python
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for xi, yi in zip(X, y):
                prediction = self._predict(xi)
                update = self.lr * (yi - prediction)
                self.weights += update * xi
                self.bias += update

    def _predict(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

    def predict(self, X):
        return [self._predict(x) for x in X]


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = [0, 0, 0, 1]  # AND gate: only true when both inputs are 1

p = Perceptron()
p.fit(X, y_and)
print("Predictions:", p.predict(X))  # [0, 0, 0, 1] ✓
```

> **Limitation:** A single perceptron can only solve **linearly separable** problems (where a single straight line separates the classes). XOR, for example, cannot be solved by one perceptron. This limitation is exactly why we stack multiple perceptrons into multi-layer networks (deep learning).

---

# Activation Functions

An activation function decides **whether and how strongly a neuron "fires"** after computing its weighted sum. Without activation functions, no matter how many layers you stack, the whole network collapses into a single linear transformation — useless for complex patterns.

**Why non-linearity matters:** The real world is non-linear. Activation functions introduce the curves and bends needed to model complex relationships.

**Analogy:** A dimmer switch (smooth control) vs. a light switch (binary on/off). Different activation functions give neurons different response profiles — some respond gradually, some sharply, some only for positive inputs.

---

## Types of Activation Functions

### Step Function (Classic Perceptron)
Binary: 0 or 1. Only used in historical perceptrons — not in modern networks.
```python
def step(x):
    return 1 if x >= 0 else 0
```
**Problem:** Zero gradient everywhere — impossible to use with gradient-based learning.

---

### Sigmoid
Squashes any input smoothly into the range (0, 1). Useful for probability outputs.
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(sigmoid(0))    # 0.5   ← uncertainty
print(sigmoid(5))    # 0.993 ← confident positive
print(sigmoid(-5))   # 0.007 ← confident negative
```
**Best for:** Output layer of binary classifiers.
**Problem:** Vanishing gradient — for very large or small inputs, gradient ≈ 0, making learning in deep networks very slow.

---

### Tanh (Hyperbolic Tangent)
Like sigmoid but outputs between -1 and 1. Zero-centered, which makes optimization easier.
```python
def tanh(x):
    return np.tanh(x)

print(tanh(0))    #  0.0
print(tanh(2))    #  0.964
print(tanh(-2))   # -0.964
```
**Better than sigmoid for hidden layers**, but still suffers from vanishing gradients at extremes.

---

### ReLU (Rectified Linear Unit)
**The default choice for hidden layers in modern networks.** Simple: pass positive values through unchanged, set negatives to 0.
```python
def relu(x):
    return max(0, x)

import numpy as np
x = np.array([-3, -1, 0, 1, 5])
print(np.maximum(0, x))  # [0, 0, 0, 1, 5]
```
**Why it dominates:** Computationally trivial, doesn't saturate for positive values, trains deep networks well.
**Problem:** "Dying ReLU" — neurons can get stuck outputting 0 forever if weights push them permanently negative.

---

### Leaky ReLU
Fixes Dying ReLU by allowing a tiny gradient for negative inputs instead of hard zero.
```python
def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

# Negative input: instead of 0, gives -0.01 * |x| — keeps the gradient alive
```

---

### Softmax
Converts a vector of raw scores (logits) into **probabilities that sum to exactly 1**. Always used in the output layer for multi-class classification.
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()

scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print(probs)           # [0.659, 0.242, 0.099]
print(probs.sum())     # 1.0
# Interpretation: 65.9% class 0, 24.2% class 1, 9.9% class 2
```

---

## Quick Reference: When to Use What

| Position | Task | Activation to use |
|----------|------|-------------------|
| Hidden layers (default) | Any deep network | **ReLU** |
| Hidden layers (deep, dying ReLU issue) | Any | **Leaky ReLU** |
| Output — binary classification | Predict yes/no probability | **Sigmoid** |
| Output — multi-class classification | Predict one of N classes | **Softmax** |
| Output — regression | Predict a number | **None (linear / identity)** |
| Recurrent networks (LSTM, GRU) | Sequential data | **Tanh** (gates use Sigmoid) |

---

# Neural Networks and Deep Learning

## Feedforward Neural Network (FNN)

The simplest neural network. Data flows in **one direction only**: input → hidden layers → output. No loops, no memory.

**Analogy:** A factory assembly line. Raw materials (input features) pass through a series of processing stations (layers), each transforming them further, until a finished product (prediction) comes out at the end.

```
Input Layer  →  Hidden Layer 1  →  Hidden Layer 2  →  Output Layer
(features)       (transforms)        (transforms)       (prediction)
```

```python
import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),    # 4 input features → 16 neurons
            nn.ReLU(),           # activation function (see below)
            nn.Linear(16, 8),    # 16 → 8 neurons
            nn.ReLU(),
            nn.Linear(8, 3)      # 8 → 3 output classes
        )

    def forward(self, x):
        return self.network(x)

model = FNN()
sample = torch.rand(1, 4)          # 1 sample, 4 features
output = model(sample)
print("Output logits:", output)    # raw scores before softmax
```

---

## Dropout and Batch Normalization

Two essential building blocks that make deep networks train better.

### Dropout
Randomly **sets a fraction of neurons to zero** during each training step. This prevents the network from relying too heavily on any single neuron ("co-adaptation") and acts as regularization.

**Analogy:** A sports team where random players sit out each practice. Everyone is forced to be well-rounded because they can't rely on the star player always being available.

### Batch Normalization
**Normalizes the inputs to each layer** (mean=0, variance=1) across the current batch. This stabilizes training, allows higher learning rates, and speeds up convergence.

**Analogy:** Recalibrating your instruments between each stage of a factory assembly line — ensures each stage receives inputs in a consistent range.

```python
import torch.nn as nn

class BetterFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),   # normalize after linear, before activation
            nn.ReLU(),
            nn.Dropout(0.3),      # randomly zero out 30% of neurons during training
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)

model = BetterFNN()
model.train()   # dropout is active
model.eval()    # dropout is disabled (use full network for inference)
```

> **Key detail:** Dropout is only active during `model.train()`. During inference (`model.eval()`), all neurons are used — their outputs are scaled accordingly.

---

## Loss Function

Measures **how wrong the model's predictions are**. The goal of all training is to minimize this number.

**Analogy:** Like a GPS telling you how far off-course you are. The model uses this error signal to correct itself after every batch of examples.

| Task | Common Loss Function | When to use |
|------|---------------------|-------------|
| Regression | MSE (Mean Squared Error) | Predicting continuous values |
| Binary Classification | Binary Cross-Entropy | Two classes (yes/no, spam/not spam) |
| Multi-class Classification | Cross-Entropy | 3+ classes (cat/dog/bird) |

```python
import torch
import torch.nn.functional as F

predictions = torch.tensor([[2.0, 1.0, 0.1]])  # raw scores (logits) for 3 classes
targets = torch.tensor([0])                    # true label = class 0

loss = F.cross_entropy(predictions, targets)
print(f"Loss: {loss.item():.4f}")
# A perfect prediction → loss near 0
# A wrong prediction → loss is large (penalizes more the more confident and wrong you are)
```

---

## Backpropagation

The algorithm that **updates the model's weights** based on the loss. This is how neural networks actually learn.

**Analogy:** You throw a dart and miss the bullseye. To improve, you trace back every micro-decision: how you held your arm, wrist angle, release point. Backpropagation does this mathematically — traces the error backward through every layer and figures out how much each weight contributed to the mistake.

**How it works step-by-step:**
1. **Forward pass** — run input through the network, get a prediction
2. **Calculate loss** — how wrong was the prediction?
3. **Backward pass** — compute gradients (partial derivatives) for every weight using the chain rule
4. **Update weights** — nudge each weight in the direction that reduces the loss (optimizer step)

```python
# A complete minimal training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()                        # 1. clear old gradients
    output = model(X_batch)                      # 2. forward pass
    loss = F.cross_entropy(output, y_batch)      # 3. compute loss
    loss.backward()                              # 4. backpropagation
    optimizer.step()                             # 5. update weights

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## Gradient Descent and Optimizers

Gradient descent is the **engine behind all neural network training** — and many classical ML algorithms too. The backpropagation section above computes the gradients; gradient descent decides what to do with them.

**Core idea:** The loss function creates a "landscape" of hills and valleys. Your model's current weights are a point on that landscape. Gradient descent figures out which direction is downhill (the gradient) and takes a step in that direction.

**Analogy:** You're blindfolded on a mountain and need to reach the valley. You feel the slope under your feet, take a step in the steepest downhill direction, feel again, step again — until you reach the bottom.

### Types of Gradient Descent

| Type | How it works | Tradeoff |
|------|-------------|----------|
| **Batch GD** | Computes gradient over the entire dataset per step | Stable but slow for large datasets |
| **Stochastic GD (SGD)** | Uses one random sample per step | Fast but noisy, can escape local minima |
| **Mini-batch GD** | Uses a small batch (e.g., 32 or 64 samples) per step | Best of both worlds — the standard approach |

### Learning Rate
The **step size** of each update. The most important hyperparameter:
- **Too high** → overshoots the minimum, loss oscillates or explodes
- **Too low** → converges painfully slowly, may get stuck
- **Just right** → steady convergence

### Common Optimizers

| Optimizer | What it adds over SGD | When to use |
|-----------|----------------------|-------------|
| **SGD** | Nothing — vanilla | Simple, well-understood, often needs LR tuning |
| **SGD + Momentum** | Remembers previous gradient direction (inertia) | Faster convergence, fewer oscillations |
| **Adam** | Adaptive per-parameter learning rates + momentum | **Default choice** — works well out of the box |
| **AdamW** | Adam + weight decay (proper L2 regularization) | Standard for training transformers and LLMs |

```python
# The existing training loop uses Adam — here's why it's the default:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Adam adjusts the learning rate for each weight individually,
# so features that update rarely get bigger steps — very practical.

# For transformers, AdamW is preferred:
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
```

> **Practical advice:** Start with **Adam** (lr=0.001) for most tasks. Use **AdamW** for fine-tuning transformers. Only switch to SGD+momentum if you need to squeeze the last bit of performance and are willing to tune aggressively.

### Vanishing and Exploding Gradients

During backpropagation, gradients are multiplied through each layer via the chain rule. In deep networks, this repeated multiplication can cause problems:

- **Vanishing gradients:** Gradients shrink exponentially toward zero as they flow backward. Early layers barely update → the network can't learn deep features. This plagued sigmoid/tanh activations and vanilla RNNs.
- **Exploding gradients:** Gradients grow exponentially, causing weights to swing wildly. Training becomes unstable (loss = NaN).

**Solutions:**
- Use **ReLU** activation (doesn't squash gradients for positive values)
- Use **proper weight initialization** (see below)
- Use **batch normalization** (stabilizes layer inputs)
- Use **residual/skip connections** (ResNet — gradient flows through shortcut paths)
- Use **gradient clipping** for exploding gradients: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### Weight Initialization

How you initialize weights before training **dramatically affects** whether training converges. All-zeros is catastrophic (all neurons learn the same thing). Too large or too small values cause exploding/vanishing gradients.

| Method | Formula | Best for |
|--------|---------|----------|
| **Xavier/Glorot** | Scales by 1/√(fan_in + fan_out) | Sigmoid, Tanh activations |
| **He/Kaiming** | Scales by √(2/fan_in) | ReLU activations (the default in PyTorch) |

PyTorch's `nn.Linear` uses Kaiming initialization by default — so you rarely need to set this manually, but knowing *why* matters for debugging.

### Learning Rate Scheduling

In practice, you almost never use a fixed learning rate. **Schedulers adjust the LR during training** for better convergence.

| Scheduler | How it works | When to use |
|-----------|-------------|-------------|
| **StepLR** | Multiply LR by γ every N epochs | Simple, predictable decay |
| **CosineAnnealingLR** | LR follows a cosine curve down to 0 | Standard for image models |
| **Warmup + Linear Decay** | Start low, ramp up, then linearly decrease | Standard for Transformer fine-tuning |
| **ReduceLROnPlateau** | Reduce LR when validation loss plateaus | Adaptive, easy to use |

```python
# Warmup + linear decay — standard for fine-tuning BERT/LLMs
from transformers import get_linear_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,    # warm up for first 100 steps
    num_training_steps=1000  # total training steps
)

for batch in dataloader:
    loss = train_step(batch)
    loss.backward()
    optimizer.step()
    scheduler.step()   # update LR after each step
```

> **Why warmup?** At the start of training, the model's random weights produce large gradients. A high learning rate + large gradients = unstable training. Warmup uses a tiny LR initially, then gradually increases it once gradients stabilize. This is critical for Transformer training.

---

## Convolutional Neural Networks (CNNs)

CNNs are the dominant architecture for **image and spatial data** tasks. Instead of connecting every input to every neuron (like FNNs), CNNs use small sliding filters that detect local patterns.

**How it works:**
1. **Convolution layers** — small filters (e.g., 3x3) slide across the image, detecting patterns like edges, textures, shapes
2. **Pooling layers** — downsample the output (e.g., take the max of each 2x2 region) to reduce size and add invariance
3. **Fully connected layers** — at the end, flatten and classify

**Analogy:** A magnifying glass scanning across a photo. Early layers detect simple features (edges, corners). Middle layers combine them into textures and shapes. Final layers recognize objects ("that's a cat").

```
Input Image → [Conv → ReLU → Pool] → [Conv → ReLU → Pool] → Flatten → Dense → Output
   (3x32x32)     (edges)                (shapes)                         (cat/dog/bird)
```

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 3 input channels (RGB) → 16 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 32x32 → 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # 16 → 32 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 16x16 → 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)                              # 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN()
sample = torch.rand(1, 3, 32, 32)  # 1 image, 3 channels (RGB), 32x32 pixels
print(model(sample).shape)          # torch.Size([1, 10]) — 10 class scores
```

**Famous CNN architectures:** LeNet (1998), AlexNet (2012, started the deep learning revolution), VGG, ResNet (skip connections), EfficientNet.

> **CNNs vs FNNs for images:** An FNN treating a 224x224 RGB image would have 150K+ input neurons with no spatial awareness. A CNN uses parameter sharing (same filter slides everywhere) — far fewer parameters and it understands spatial structure.

---

## Recurrent Neural Networks (RNNs), LSTMs, GRUs

RNNs are designed for **sequential data** — text, time series, audio — where order matters and past context informs the present.

**Core idea:** Unlike FNNs (input → output), RNNs have a **hidden state** that carries information from previous steps, creating a form of memory.

**Analogy:** Reading a sentence word by word. After each word, you update your mental understanding (hidden state). By the time you reach the period, your understanding reflects the entire sentence.

### Vanilla RNN — The Problem
Simple RNNs suffer from the **vanishing gradient problem**: when sequences are long, gradients shrink to near-zero during backpropagation, making it impossible to learn long-range dependencies.

### LSTM (Long Short-Term Memory)
Solves vanishing gradients with **three gates** that control information flow:

| Gate | What it does |
|------|-------------|
| **Forget gate** | Decides what to discard from memory |
| **Input gate** | Decides what new information to store |
| **Output gate** | Decides what to output from memory |

### GRU (Gated Recurrent Unit)
A simplified LSTM with **two gates** (reset and update). Fewer parameters, similar performance, faster to train.

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)          # tokens → vectors
        _, (hidden, _) = self.lstm(embedded)  # process sequence, get final hidden state
        return self.fc(hidden.squeeze(0))     # classify using final state

model = LSTMClassifier(vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=2)
sample = torch.randint(0, 10000, (1, 50))  # 1 sentence, 50 tokens
print(model(sample).shape)                  # torch.Size([1, 2]) — binary classification
```

> **RNNs vs Transformers:** Transformers have largely replaced RNNs for most NLP tasks because they process all tokens in parallel (faster) and handle long-range dependencies better via attention. RNNs are still used in streaming/real-time applications where you process one token at a time.

---

# Parameters and Hyperparameters

## Parameters — Learned by the Model

Parameters are the **internal numerical values the model learns during training** — primarily weights and biases. You never set them manually; the optimizer adjusts them automatically to minimize the loss.

**Analogy:** Baking a cake. Parameters are the exact amounts of each ingredient (2.3 cups of flour, 1.1 cups of sugar) that the chef refines automatically after tasting each trial. They're adjusted internally, not chosen upfront.

| Example | What it represents |
|---------|-------------------|
| Weights in a neural network | How much each input contributes to each neuron's output |
| Bias terms | Shifts the neuron's activation threshold |
| Coefficients in linear regression | Slope and intercept of the fitted line |

```python
from sklearn.linear_model import LinearRegression

X = [[1000], [1500], [2000], [2500]]
y = [200000, 280000, 350000, 420000]

model = LinearRegression()
model.fit(X, y)

# These were learned automatically — you didn't set them
print("Learned weight (slope):", model.coef_)       # e.g. [150.0]
print("Learned bias (intercept):", model.intercept_) # e.g. 50000.0
```

---

## Hyperparameters — Set by You Before Training

Hyperparameters are the **configuration settings you choose before training begins**. They control *how* training happens, not the model's weights.

**Analogy:** Still baking that cake — hyperparameters are the **oven temperature** and **baking time**. You set them before you start. They determine whether the model trains well, but they don't get adjusted automatically during training.

| Hyperparameter | What it controls |
|----------------|-----------------|
| **Learning rate** | Size of each weight update step — too high = overshooting, too low = slow |
| **Epochs** | How many full passes over the training data |
| **Batch size** | How many samples before each weight update |
| **Number of layers** | Depth of the network |
| **Hidden layer size** | Width (capacity) of each layer |
| **Dropout rate** | Fraction of neurons randomly disabled during training (prevents overfitting) |
| **n_estimators** | Number of trees in Random Forest / XGBoost |
| **k** (KNN) | Number of nearest neighbors to consider |
| **C** (SVM) | Regularization strength (larger C = fit training data tighter) |
| **max_depth** (Tree) | Maximum depth of a decision tree |

```python
from sklearn.ensemble import GradientBoostingClassifier

# All of these are hyperparameters — chosen by you, not learned
model = GradientBoostingClassifier(
    n_estimators=200,       # number of trees
    learning_rate=0.05,     # how aggressively to fit each new tree
    max_depth=4,            # max depth per tree
    min_samples_split=10,   # min samples required to split a node
    subsample=0.8,          # use 80% of data per tree (reduces overfitting)
    random_state=42
)
model.fit(X_train, y_train)
```

---

## Hyperparameter Tuning

Finding the best hyperparameters is itself a process called **hyperparameter search** or **model selection**:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the search space
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Try all combinations, evaluate each with 5-fold cross-validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # use all CPU cores
)
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best CV accuracy:", f"{grid_search.best_score_:.2%}")
```

**Other tuning strategies:**
- **Grid Search** — try every combination (exhaustive, slow for large spaces)
- **Random Search** — sample random combinations (faster, surprisingly effective)
- **Bayesian Optimization** (Optuna, Hyperopt) — intelligently narrows the search based on previous results (best for large search spaces)

> **Golden rule:** Always tune hyperparameters using a **validation set** or **cross-validation**, never on the test set — otherwise you're effectively training on your test set and your evaluation metric becomes meaningless.

---

---
---

# PART 6 — REPRESENTATIONS & ATTENTION

---

# From One-Hot to Embeddings — How Words Become Numbers

## One-Hot Encoding (the naive approach)
Represent each word as a vector with a 1 in one position and 0s everywhere else. If your vocabulary has 50,000 words, each word is a 50,000-dimensional vector.

```
"cat"  = [1, 0, 0, 0, ...]   (50,000 dimensions)
"dog"  = [0, 1, 0, 0, ...]
"king" = [0, 0, 1, 0, ...]
```

**Problems:**
- **Massive and sparse** — 50K-dimensional vector with a single 1
- **No semantic relationship** — "cat" and "dog" are as different as "cat" and "earthquake" (distance is identical)
- **Doesn't scale** — vocabulary growth = dimension explosion

One-hot encoding is still used for categorical features in tabular ML (e.g., `color: [red, blue, green]` → `[1,0,0], [0,1,0], [0,0,1]`), but it's terrible for representing words in NLP.

---

# Word Embeddings (Word2Vec, GloVe)

Word embeddings represent words as **dense numerical vectors** where semantic meaning is captured by position in space. Similar words land near each other.

**The revolution:** Before embeddings, words were represented as one-hot vectors (sparse, no semantic relationship). "king" and "queen" were as different as "king" and "banana." Embeddings changed that — they encode meaning.

**The famous example:**
```
king - man + woman ≈ queen
paris - france + italy ≈ rome
```
Vector arithmetic captures semantic relationships!

**Analogy:** Imagine plotting every word in the dictionary on a giant map. Words with similar meanings cluster together. Verbs in one region, animals in another, countries in another. Embeddings ARE the coordinates of each word on this map.

## Word2Vec (Google, 2013)

Two training approaches:
- **CBOW** (Continuous Bag of Words): predict the center word from surrounding words
- **Skip-gram**: predict surrounding words from the center word

## GloVe (Stanford, 2014)

Uses a co-occurrence matrix — counts how often words appear near each other across the entire corpus, then factorizes the matrix into vectors.

```python
import gensim.downloader as api

# Load pre-trained Word2Vec (trained on Google News, 3M words)
model = api.load("word2vec-google-news-300")

# Find similar words
print(model.most_similar("king", topn=5))
# [('kings', 0.71), ('queen', 0.65), ('monarch', 0.64), ...]

# Vector arithmetic
result = model.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
print(result)  # [('queen', 0.71)]

# Word similarity
print(model.similarity("cat", "dog"))     # ~0.76 (related)
print(model.similarity("cat", "rocket"))  # ~0.08 (unrelated)
```

> **Limitation:** Word2Vec and GloVe give each word a **single** fixed vector regardless of context. "bank" (river) and "bank" (financial) get the same embedding. This is solved by **contextual embeddings** (BERT, GPT), where the embedding changes based on surrounding words.

---

# Attention Mechanism

The attention mechanism lets a model **dynamically focus on the most relevant parts of the input** when producing each part of the output. It's the core innovation behind Transformers.

**The problem it solves:** In RNNs, information from early in a sequence gets "diluted" as the sequence gets longer (vanishing gradient). Attention allows the model to directly look at any position, regardless of distance.

**Analogy:** Reading a long legal document to answer a specific question. Instead of reading every word equally, you scan the document and focus intensely on the relevant paragraphs while skimming the rest. That's attention — dynamically allocating focus based on relevance.

## Self-Attention (the heart of Transformers)

Every token attends to every other token in the sequence, computing how relevant each one is to each other.

### The Query-Key-Value (Q, K, V) framework:

**Analogy:** A library search.
- **Query (Q):** "I'm looking for information about climate change" (what you want)
- **Key (K):** The label/title of each book on the shelf (what each item offers)
- **Value (V):** The actual content of each book (the information to retrieve)

Attention = match queries against keys to determine relevance, then retrieve a weighted mix of values.

### The formula (in plain terms):
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

1. Q × K^T       → compute relevance scores (dot product between query and each key)
2. / √d           → scale down to prevent extreme values (d = dimension of keys)
3. softmax(...)   → convert to probabilities (how much to attend to each position)
4. × V           → weighted sum of values based on attention weights
```

```python
import torch
import torch.nn.functional as F

# Simplified self-attention for 4 tokens, each with 8-dimensional embeddings
d_k = 8
seq_len = 4

Q = torch.rand(seq_len, d_k)  # queries
K = torch.rand(seq_len, d_k)  # keys
V = torch.rand(seq_len, d_k)  # values

# Scaled dot-product attention
scores = Q @ K.T / (d_k ** 0.5)       # (4, 4) — each token's relevance to every other
weights = F.softmax(scores, dim=-1)    # normalize to probabilities
output = weights @ V                    # weighted sum of values

print("Attention weights:\n", weights)  # each row shows how much token i attends to each token j
print("Output shape:", output.shape)    # (4, 8) — each token gets a context-aware representation
```

### Multi-Head Attention
Instead of computing attention once, Transformers run **multiple attention heads in parallel** — each head can learn to attend to different types of relationships (syntactic, semantic, positional, etc.).

```
Multi-Head Attention with 8 heads:
  Head 1 → learns subject-verb relationships
  Head 2 → learns adjective-noun relationships
  Head 3 → learns coreference (pronouns → nouns they refer to)
  ...
  → All 8 outputs concatenated and projected → final output
```

### Positional Encoding

Attention has no concept of word order — "The cat sat on the mat" and "mat the on sat cat the" would produce identical attention patterns. **Positional encoding** injects position information into the input embeddings.

**How it works:** Add a unique vector to each token's embedding that encodes its position in the sequence. The original Transformer uses sine/cosine functions at different frequencies:

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
    return pe

# Each position gets a unique pattern of sine/cosine values
# Position 0 → [sin(0), cos(0), sin(0), cos(0), ...]
# Position 1 → [sin(1), cos(1), sin(0.01), cos(0.01), ...]
# The model learns to use these patterns to understand word order
```

**Modern alternatives:** BERT uses learned positional embeddings (a trainable vector per position). RoPE (Rotary Position Embeddings) is used by Llama and most modern LLMs — it encodes relative positions and supports extending context length beyond training.

> **Why attention dominates:** RNNs process tokens one-by-one (sequential, slow). Attention processes ALL tokens simultaneously (parallelizable, fast on GPUs) and captures long-range dependencies equally well regardless of distance.

---

# The Transformer Architecture ("Attention Is All You Need")

The Transformer (2017) is the architecture behind BERT, GPT, Claude, and virtually all modern AI. Understanding its building blocks is essential.

## Full Architecture (each block stacked N times):

```
INPUT TOKENS
     ↓
[Token Embeddings + Positional Encoding]
     ↓
┌─────────────────────────────┐
│     Multi-Head Attention     │  ← each token attends to all others
│            ↓                 │
│     Add & Layer Norm         │  ← residual connection + normalization
│            ↓                 │
│     Feed-Forward Network     │  ← 2 linear layers with ReLU: expand then compress
│            ↓                 │
│     Add & Layer Norm         │  ← another residual connection
└─────────────────────────────┘
     ↓  (repeat N times)
   OUTPUT
```

## Key building blocks explained:

| Component | What it does | Why it matters |
|-----------|-------------|---------------|
| **Token Embeddings** | Convert token IDs to dense vectors | Words become numbers the model can process |
| **Positional Encoding** | Add position information | Without this, word order is lost |
| **Multi-Head Self-Attention** | Each token attends to every other token | Captures context and dependencies |
| **Residual Connections (Add)** | Add the input of a sub-layer to its output | Enables training very deep networks (gradient flows through shortcuts) |
| **Layer Normalization** | Normalize across features for each token | Stabilizes training, speeds up convergence |
| **Feed-Forward Network** | Two linear layers (expand → ReLU → compress) | Adds non-linear transformation capacity per-token |

## How the original paper uses Encoder + Decoder:

The original "Attention Is All You Need" paper has both an encoder and decoder stack (for machine translation). Each decoder layer has an additional **cross-attention** sub-layer that attends to the encoder's output.

```
Encoder (6 layers):  Self-Attention → Feed-Forward  (processes input)
Decoder (6 layers):  Masked Self-Attention → Cross-Attention → Feed-Forward  (generates output)
                     ↑ can only see past tokens    ↑ attends to encoder
```

**Masked self-attention** in the decoder prevents tokens from attending to future positions — ensuring the model can only use information from tokens it has already generated (autoregressive).

> **Why the Transformer won:** It's parallelizable (unlike RNNs), handles long-range dependencies (unlike CNNs), and scales efficiently to billions of parameters. These properties made GPT-3, GPT-4, Claude, and all modern LLMs possible.

---

# Encoder-Decoder Architecture

The encoder-decoder is a fundamental architecture pattern for tasks that **transform one sequence into another** (translation, summarization, text-to-code).

**Analogy:** A translator at the UN. The **encoder** is the interpreter who listens to the full speech in French and forms a mental understanding. The **decoder** is the interpreter who produces the translation in English from that understanding.

## How it works:
1. **Encoder** processes the full input and compresses it into a rich internal representation (context)
2. **Decoder** generates the output step-by-step, attending to the encoder's representation

## Three architecture patterns in modern AI:

| Architecture | How it works | Example models | Best for |
|-------------|-------------|----------------|----------|
| **Encoder-only** | Processes full input bidirectionally; outputs a representation | BERT, RoBERTa | Understanding: classification, NER, embeddings |
| **Decoder-only** | Generates tokens left-to-right, autoregressively | GPT, Claude, Llama | Generation: chatbots, code, creative writing |
| **Encoder-Decoder** | Encoder reads input, decoder generates output | T5, BART, Whisper | Transformation: translation, summarization |

```
Encoder-only (BERT):     [Input] → [Encoder] → Understanding/Classification

Decoder-only (GPT):      [Prompt] → [Decoder → token → token → token] → Generated text

Encoder-Decoder (T5):    [Input] → [Encoder] → [Decoder → token → token] → Transformed output
```

> **Why this matters:** Knowing the architecture type tells you what a model is good at. Don't use BERT (encoder-only) for text generation. Don't use GPT (decoder-only) when you need deep bidirectional understanding of a passage.

---

---
---

# PART 7 — TRANSFORMER MODELS

---

# BERT

**BERT** (Bidirectional Encoder Representations from Transformers) was released by Google in 2018. It redefined NLP benchmarks across the board and remains highly useful today — especially when a full LLM is overkill, too slow, or too expensive.

## BERT vs GPT — Key Difference

| | BERT | GPT |
|---|------|-----|
| **Direction** | Bidirectional (reads left AND right simultaneously) | Left-to-right only (autoregressive) |
| **Strength** | *Understanding* text | *Generating* text |
| **Pretraining** | Masked Language Modeling + NSP | Next token prediction |
| **Best for** | Classification, NER, embeddings, QA | Chatbots, code gen, creative writing |

**Analogy:** BERT reads a sentence the way a human does — taking in the full context before interpreting any word. GPT reads like a typewriter — one word at a time, left to right, never peeking ahead.

---

## BERT Pretraining Tasks

BERT was pretrained on two self-supervised tasks on a massive text corpus (Wikipedia + BookCorpus):

### Masked Language Modeling (MLM)
15% of words are randomly masked; BERT must predict them using surrounding context from both directions.
```
Input:  "The [MASK] sat on the [MASK]."
Output: "The cat  sat on the mat."
```
This forces BERT to build deep bidirectional understanding of language.

### Next Sentence Prediction (NSP)
Given two sentences A and B, predict whether B is the actual next sentence after A in the original text.
```
A: "The dog barked loudly."
B: "It was trying to scare the mailman."   → IsNext: True  ✓

A: "The dog barked loudly."
B: "Quantum physics is fascinating."       → IsNext: False ✗
```

---

## Fine-tuning BERT

After pretraining, BERT can be fine-tuned on any specific downstream task by adding a small task-specific output head and training on labeled data:

| Task | What is added on top of BERT |
|------|------------------------------|
| **NER** | Token classifier — labels each word with an entity type |
| **Question Answering** | Span predictor — finds start and end of answer in context |
| **Summarization** | Sequence-to-sequence decoder |
| **Sentiment Analysis** | Sequence classifier using [CLS] token |
| **Embeddings / Feature extraction** | Use [CLS] token or mean-pool layer outputs directly |
| **Sentence Pair Tasks** | Feed both sentences, classify relationship (entailment, contradiction, neutral) |

---

## BERT Model Sizes

| Model | Transformer Layers | Hidden Size | Attention Heads | Parameters |
|-------|-------------------|-------------|-----------------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

---

## BERT Variants

| Variant | What's different |
|---------|----------------|
| **RoBERTa** | Improved BERT: more data, longer training, no NSP — usually outperforms BERT |
| **DistilBERT** | 40% smaller, 60% faster, retains ~97% of BERT performance — great for production |
| **ALBERT** | Parameter sharing to dramatically reduce model size |
| **BioBERT** | Pretrained on biomedical text (PubMed) — better for clinical/scientific NLP |
| **FinBERT** | Pretrained on financial text — better for finance sentiment, reports |
| **XLM-RoBERTa** | Multilingual, supports 100 languages |

---

---

# GPT (Generative Pre-trained Transformer)

**GPT** is the decoder-only transformer family behind ChatGPT, the model that brought AI to mainstream consciousness. While BERT excels at understanding, GPT excels at **generating** text.

## How GPT Works

GPT is trained with a single, elegant objective: **predict the next token** given all previous tokens (autoregressive language modeling).

```
Input:  "The capital of France is"
GPT predicts next token probabilities:
  "Paris" → 0.92
  "Lyon"  → 0.03
  "a"     → 0.01
  ...
```

During training, GPT reads enormous amounts of text and learns to predict each next word. This simple task, at massive scale, teaches the model grammar, facts, reasoning, code, and more.

## GPT Evolution

| Model | Year | Parameters | Key innovation |
|-------|------|-----------|----------------|
| GPT-1 | 2018 | 117M | Showed pre-training + fine-tuning works for NLP |
| GPT-2 | 2019 | 1.5B | Zero-shot task performance emerges at scale |
| GPT-3 | 2020 | 175B | Few-shot learning via prompting (no fine-tuning needed) |
| GPT-3.5 / ChatGPT | 2022 | ~175B | RLHF alignment → first mainstream AI chatbot |
| GPT-4 | 2023 | ~1.8T (MoE) | Multimodal (text + images), dramatically improved reasoning |

## BERT vs GPT — When to Use Which

| Scenario | Use BERT (encoder) | Use GPT (decoder) |
|----------|-------------------|-------------------|
| Classify emails as spam | Yes — needs understanding | Overkill |
| Extract entities from documents | Yes — bidirectional context | Can do it, but slower |
| Build a chatbot | No — can't generate text | Yes — designed for this |
| Summarize articles | Not ideal | Yes — generates summaries |
| Compute text embeddings | Yes — efficient, purpose-built | Can but SBERT is better |
| Answer open-ended questions | Limited | Yes — excels at this |

> **In production:** Most companies use BERT-family models (or SBERT) for search, classification, and embeddings. They use GPT-family models (or Claude) for content generation, chatbots, code assistants, and complex reasoning. Many systems use BOTH: BERT for fast retrieval, then GPT for generation (this is RAG).

---

## Example: BERT for Sentiment Analysis

```python
from transformers import pipeline

# HuggingFace downloads and caches the fine-tuned model automatically
sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

texts = [
    "This product is absolutely amazing!",
    "Terrible quality, complete waste of money.",
    "It arrived on time and works as described."
]

for text in texts:
    result = sentiment(text)[0]
    print(f"Text: {text}")
    print(f"  → {result['label']} (confidence: {result['score']:.2%})\n")
```

---

# Sentence Transformers (SBERT)

**SBERT** (Sentence-BERT) is built on top of BERT but specifically trained to produce a **single fixed-size vector (embedding) for an entire sentence** — efficiently and semantically.

**The problem SBERT solves:** Vanilla BERT computes similarity by passing *both* sentences through together, which is slow. Comparing 10,000 sentences against each other would require ~50 million BERT passes. SBERT computes one embedding per sentence upfront, then uses fast cosine similarity — reducing 50M passes to 10K + vector math.

**Analogy:** Instead of reading two books side-by-side every time you want to compare them (BERT), SBERT gives each book a unique "fingerprint" (embedding vector). Comparing fingerprints takes microseconds.

## When is SBERT useful?

| Use Case | How SBERT helps |
|----------|----------------|
| **Semantic search** | Embed query + docs, find most similar by cosine similarity |
| **Document clustering** | Group similar texts together by embedding proximity |
| **Duplicate detection** | Find near-identical sentences across a large corpus |
| **Recommendation systems** | "Similar items" based on text description embeddings |
| **RAG (Retrieval-Augmented Generation)** | Retrieve relevant chunks before sending to LLM |

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')  # fast, lightweight, effective

sentences = [
    "A man is eating a pizza.",
    "Someone is having Italian food.",
    "The cat sat on the mat.",
]

embeddings = model.encode(sentences)  # each sentence → 384-dim vector

sim_1_2 = util.cos_sim(embeddings[0], embeddings[1])
sim_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print(f"Pizza vs Italian food:  {sim_1_2.item():.4f}")  # high similarity (~0.75)
print(f"Pizza vs cat on mat:    {sim_1_3.item():.4f}")  # low similarity  (~0.05)
```

---

# Dimensionality Reduction (PCA, t-SNE)

Dimensionality reduction transforms high-dimensional data into **fewer dimensions** while preserving the most important structure. Useful for visualization, noise removal, and speeding up downstream models.

**Analogy:** You have a spreadsheet with 500 columns. Most of them are correlated or redundant. Dimensionality reduction boils it down to the 10 most informative "super-columns" — or even 2 for plotting on a chart.

## PCA (Principal Component Analysis)

Finds the **directions of maximum variance** in the data and projects onto them. Linear, fast, interpretable.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)  # 4 features

# Reduce 4 dimensions → 2 for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

print(f"Variance explained: {pca.explained_variance_ratio_}")
# e.g., [0.92, 0.05] → first 2 components explain 97% of variance

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis')
plt.xlabel("PC1"), plt.ylabel("PC2")
plt.title("Iris Dataset — PCA")
plt.show()
```

## t-SNE

Non-linear technique optimized for **visualization** of high-dimensional data in 2D/3D. Groups similar points together.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title("Iris Dataset — t-SNE")
plt.show()
```

| | PCA | t-SNE |
|---|-----|-------|
| **Type** | Linear | Non-linear |
| **Speed** | Fast | Slow |
| **Use for** | Feature reduction, preprocessing | Visualization only |
| **Preserves** | Global structure (variance) | Local structure (clusters) |

---

# Feature Engineering

Feature engineering is the process of **creating new features from existing data** to help the model learn better patterns. Often has more impact on performance than choosing a fancier algorithm.

**Analogy:** Giving the model better ingredients to cook with. Raw ingredients (dates, text, numbers) often need chopping, combining, and seasoning before they're useful.

```python
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.to_datetime(['2024-01-15 08:30', '2024-07-04 14:00', '2024-12-25 22:15']),
    'price': [10.5, 25.0, 8.0],
    'quantity': [3, 1, 5]
})

# Extract date features — models can't read dates directly
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Create interaction features
df['total_spent'] = df['price'] * df['quantity']

# Binning continuous features
df['price_tier'] = pd.cut(df['price'], bins=[0, 10, 20, 100], labels=['low', 'mid', 'high'])

print(df)
```

**Common feature engineering patterns:**
- **Date/time** → hour, day of week, month, is_holiday, time_since_event
- **Text** → word count, character count, average word length, sentiment score
- **Numerical** → log transform (for skewed data), ratios between features, polynomial features
- **Categorical** → one-hot encoding, target encoding, frequency encoding

> **When to skip it:** Deep learning (CNNs, Transformers) automates feature engineering internally — that's a major reason it outperforms classical ML on raw images and text. But for tabular data, manual feature engineering still matters.

---

---
---

# PART 8 — MODERN GenAI

---

# Large Language Models (LLMs)

LLMs are **massive transformer-based models** trained on internet-scale text data to predict the next token. Through sheer scale (billions of parameters, trillions of training tokens), they develop surprising general capabilities.

**What makes them "large":**

| Model | Parameters | Training data |
|-------|-----------|---------------|
| GPT-2 (2019) | 1.5B | 40GB text |
| GPT-4 (2023) | ~1.8T (estimated) | ~13T tokens |
| Claude 3.5 Sonnet | Undisclosed | Undisclosed |
| Llama 3 (2024) | 8B - 405B | 15T tokens |

**How they work:** At their core, LLMs do one thing — **predict the most likely next token** given all previous tokens. But this simple objective, at scale, produces emergent abilities:
- Reasoning and problem-solving
- Code generation
- Translation between languages
- Summarization
- Following complex instructions

**Analogy:** An LLM is like someone who has read the entire internet and can continue any sentence you start. Ask it to continue "The capital of France is..." and it says "Paris." Ask it to continue "def fibonacci(n):\n" and it writes code. The "intelligence" emerges from learning patterns across enormous amounts of text.

> **Key insight:** LLMs don't "know" things the way humans do. They are statistical pattern-completion machines. This is why they can be confidently wrong (hallucination) — they generate what *sounds* plausible, not what *is* true.

---

# Tokens and Context Window

## Tokens — What LLMs Actually Process

LLMs don't read words or characters — they process **tokens**, which are subword pieces (see LLM Tokenization above). Everything the model sees, generates, and counts is in tokens.

**Rough conversion:** In English, 1 token ≈ 0.75 words ≈ 4 characters.

```python
import tiktoken  # OpenAI's tokenizer

enc = tiktoken.encoding_for_model("gpt-4")

text = "Machine learning is fascinating and powerful!"
tokens = enc.encode(text)

print(f"Text: '{text}'")
print(f"Words: {len(text.split())}")    # 6 words
print(f"Tokens: {len(tokens)}")          # 7 tokens
print(f"Token IDs: {tokens}")
print(f"Decoded: {[enc.decode([t]) for t in tokens]}")
# ['Machine', ' learning', ' is', ' fascinating', ' and', ' powerful', '!']
```

## Context Window — The Model's Working Memory

The context window is the **maximum number of tokens the model can "see" at once** — including both your input (prompt) and its output (response).

| Model | Context Window |
|-------|---------------|
| GPT-3.5 | 4K / 16K tokens |
| GPT-4 | 8K / 128K tokens |
| Claude 3.5 Sonnet | 200K tokens (~150K words) |
| Llama 3 | 8K / 128K tokens |

**Practical implications:**
- Longer context = can process larger documents, longer conversations
- Everything outside the window is "forgotten"
- Both input AND output count toward the limit
- Longer contexts cost more (compute scales quadratically with attention)

> **Analogy:** The context window is like a desk. A 4K-token desk can hold a few pages. A 200K-token desk can hold an entire book. But anything that falls off the desk is gone.

---

# Temperature, Top-k, Top-p Sampling

When an LLM generates text, it predicts a **probability distribution over all possible next tokens**. These parameters control **how the model samples from that distribution** — i.e., how "creative" vs. "focused" the output is.

## Temperature

Controls the **sharpness** of the probability distribution.

| Temperature | Effect | Use when... |
|-------------|--------|-------------|
| **0** | Always picks the highest-probability token (deterministic) | Code, math, factual answers |
| **0.3-0.7** | Mostly picks likely tokens, some variety | General conversation, summarization |
| **1.0** | Uses the natural distribution | Creative writing, brainstorming |
| **>1.0** | Flattens distribution, more random | Experimental, often incoherent |

**Analogy:** Ordering food at a restaurant.
- Temperature 0 = always order your #1 favorite dish
- Temperature 0.5 = usually your favorite, occasionally try something new
- Temperature 1.0 = randomly pick from the whole menu proportional to preference
- Temperature 2.0 = spin a wheel, anything goes

## Top-k Sampling
Only consider the **k most probable** next tokens, ignore the rest.
- Top-k=50 → choose from the 50 most likely tokens
- Cuts off the "long tail" of unlikely tokens

## Top-p (Nucleus) Sampling
Only consider tokens whose **cumulative probability reaches p**.
- Top-p=0.9 → pick from the smallest set of tokens that together have 90% probability
- Adaptive: for confident predictions, considers fewer tokens; for uncertain ones, considers more

```python
import anthropic

client = anthropic.Anthropic()

# Deterministic output (factual)
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    temperature=0,
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
print("Temp 0:", response.content[0].text)  # Always "Paris" — deterministic

# Creative output
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    temperature=1.0,
    messages=[{"role": "user", "content": "Write a one-line poem about the ocean."}]
)
print("Temp 1:", response.content[0].text)  # Different every time — creative
```

> **Best practice:** Use `temperature=0` for tasks with a "correct" answer (code, math, extraction). Use `temperature=0.5-1.0` for open-ended tasks (writing, brainstorming).

---

# Prompt Engineering

Prompt engineering is the art of **crafting inputs to get better outputs** from LLMs. How you phrase your request dramatically affects the quality of the response.

**Analogy:** Giving instructions to a brilliant but literal assistant. "Write code" gets generic output. "Write a Python function that validates email addresses using regex, with docstring and type hints, returning True/False" gets exactly what you need.

## Core Techniques

### Zero-Shot Prompting
Give the model a task with **no examples** — rely on its training knowledge.
```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": "Classify this review as positive or negative: 'The food was cold and the waiter was rude.'"}]
)
# → "Negative"
```

### Few-Shot Prompting
Provide **examples** of input-output pairs before the actual task. Dramatically improves accuracy on structured tasks.
```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=100,
    messages=[{"role": "user", "content": """Classify each review as positive or negative.

Review: "Loved the atmosphere and great service!" → positive
Review: "Overpriced and underwhelming." → negative
Review: "Best pizza I've ever had!" → positive

Review: "The pasta was soggy and tasteless." →"""}]
)
# → "negative" (model learned the pattern from examples)
```

### Chain-of-Thought (CoT)
Ask the model to **think step-by-step** before answering. Significantly improves reasoning.
```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    messages=[{"role": "user", "content": """A store sells apples at $2 each. They have a "buy 3 get 1 free" deal.
How much does it cost to buy 10 apples?

Think step by step."""}]
)
# The model reasons through the discount calculation instead of guessing
```

### System Prompts
Set the model's **role and behavior** before the conversation begins.
```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    system="You are a senior Python developer. Give concise, production-ready code. No explanations unless asked.",
    messages=[{"role": "user", "content": "Write a function to retry API calls with exponential backoff."}]
)
```

**Prompt engineering best practices:**
- Be specific about the desired format (JSON, bullet points, table)
- Provide examples when the task is structured or ambiguous
- Use "think step by step" for reasoning tasks
- Specify constraints ("in under 50 words", "using only standard library")

---

# Structured Output (JSON Mode)

In production, you almost never want free-form text from an LLM. You want **structured, parseable output** — JSON, XML, or data that feeds directly into your application code.

**Why it matters:** If your LLM returns "The sentiment is positive with a score of 0.8" as free text, you need fragile regex to extract the data. If it returns `{"sentiment": "positive", "score": 0.8}`, you just parse JSON.

```python
import anthropic
import json

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    system="You are a data extraction API. Always respond with valid JSON only, no other text.",
    messages=[{
        "role": "user",
        "content": """Extract the following from this text:
        "John Smith, age 34, works at Google in Mountain View. His email is john@google.com."

        Return JSON with fields: name, age, company, city, email"""
    }]
)

data = json.loads(response.content[0].text)
print(data)
# {"name": "John Smith", "age": 34, "company": "Google", "city": "Mountain View", "email": "john@google.com"}

# Now you can use it in your application:
save_to_database(data['name'], data['email'])
```

**In production:** Structured output is used for:
- Data extraction pipelines (parsing invoices, contracts, resumes)
- API responses from LLM-powered backends
- Automated workflows where downstream code needs predictable formats
- Classification tasks where you need a label, not a paragraph

> **Tip:** If the LLM sometimes returns invalid JSON, use libraries like `instructor` (Python) or `zod` (TypeScript) that validate and retry automatically. For Claude, you can also prefill the assistant response with `{` to force JSON output.

---

# Hallucination

Hallucination is when an LLM **confidently generates false, fabricated, or unsupported information** — presenting it as fact.

**Why it happens:** LLMs predict what *sounds plausible* based on patterns in training data, not what *is true*. They have no internal fact-checker, no database of verified knowledge, and no awareness of what they know vs. don't know.

**Analogy:** A very articulate person who has read millions of books but sometimes confuses details and fills gaps with plausible-sounding fiction — and does so with complete confidence.

## Common types of hallucination:

| Type | Example |
|------|---------|
| **Fabricated facts** | "Einstein won the Nobel Prize in 1923" (it was 1921) |
| **Invented citations** | Referencing papers, URLs, or books that don't exist |
| **Confident nonsense** | Generating plausible-sounding but incorrect code or math |
| **Conflation** | Mixing up attributes of similar entities |

## Mitigation strategies:

| Strategy | How it helps |
|----------|-------------|
| **RAG** (Retrieval-Augmented Generation) | Ground responses in real retrieved documents |
| **Lower temperature** | More deterministic → fewer creative fabrications |
| **Ask for citations** | Forces the model to attribute claims (but verify them!) |
| **Fact-check critical outputs** | Never blindly trust LLM outputs for factual claims |
| **Structured output** | JSON/schema constraints reduce free-form hallucination |

> **Golden rule:** LLMs are excellent reasoning and language tools, but they are NOT databases. Always verify factual claims, especially for medical, legal, financial, or safety-critical applications.

---

# Transfer Learning and Fine-tuning

## Transfer Learning

Transfer learning **reuses knowledge from a model trained on one task** to improve performance on a different (usually related) task. Instead of training from scratch, you start with a model that already understands general patterns.

**Analogy:** A French-trained chef learning Italian cuisine. They don't start from zero — knife skills, flavor theory, and timing all transfer. They just need to learn Italian-specific dishes.

**Why it works:** Early layers in neural networks learn general features (edges in images, grammar in text) that are universal. Only the final layers need to be specialized for your task.

## Fine-tuning

Fine-tuning **takes a pre-trained model and continues training it on your specific dataset**. The most common form of transfer learning.

## When to Use What

| Approach | When to choose | Effort | Cost |
|----------|---------------|--------|------|
| **Prompt engineering** | Quick tasks, no training data needed | Minutes | API costs only |
| **RAG** | Need access to specific/current documents | Hours | Moderate (embeddings + vector DB) |
| **Fine-tuning** | Need specialized behavior, style, or domain knowledge | Days | High (GPU compute + data collection) |

```
Task complexity →

Prompt Engineering ──→ RAG ──→ Fine-tuning ──→ Train from scratch
   (easiest)                                        (hardest)
```

> **Rule of thumb:** Try prompt engineering first. If that's not enough, try RAG. Only fine-tune when you need the model to consistently behave differently than its base behavior — specialized tone, domain expertise, or structured output format.

---

# LoRA and QLoRA

## LoRA (Low-Rank Adaptation)

LoRA is a **parameter-efficient fine-tuning** technique. Instead of updating all billions of weights in a model, it injects small trainable matrices into each layer — dramatically reducing memory and compute.

**How it works:** For a weight matrix W in the model:
- Freeze W entirely (don't update it)
- Add two small matrices A and B where W_new = W + A × B
- Only train A and B (e.g., rank 16 → adds <1% new parameters)

**Analogy:** Instead of remodeling your entire house to change the style, you add smart adapters to existing fixtures — same structure, different behavior, fraction of the cost.

## QLoRA

QLoRA = LoRA on top of a **quantized (4-bit) model**. This lets you fine-tune a 70B-parameter model on a single consumer GPU.

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA — only ~0.5% of parameters are trainable
lora_config = LoraConfig(
    r=16,                # rank of the adaptation matrices
    lora_alpha=32,       # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers to adapt
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
```

| | Full Fine-tuning | LoRA | QLoRA |
|---|---|---|---|
| **Trainable params** | All (100%) | ~0.1-1% | ~0.1-1% |
| **GPU memory (7B model)** | ~28GB | ~14GB | ~6GB |
| **Training time** | Baseline | 2-3x faster | 2-3x faster |
| **Quality** | Best | Very close | Close |

---

# RLHF (Reinforcement Learning from Human Feedback)

RLHF is the process that transforms a raw language model into an **aligned assistant** that follows instructions, refuses harmful requests, and produces helpful responses. It's how models like ChatGPT, Claude, and Gemini go from "predict next token" to "helpful AI assistant."

**The three-step process:**

### Step 1: Supervised Fine-Tuning (SFT)
Train the base model on high-quality (prompt, response) pairs written by humans. This teaches the model the *format* of helpful responses.

### Step 2: Reward Model Training
Collect human rankings: show humans multiple model responses to the same prompt and have them rank from best to worst. Train a separate model to predict these human preferences.

### Step 3: PPO (Proximal Policy Optimization)
Use the reward model as a scoring function. The LLM generates responses, the reward model scores them, and the LLM is updated via reinforcement learning to maximize the reward score — while staying close to the SFT model (to prevent reward hacking).

```
Base LLM  →  SFT (learn format)  →  RLHF (learn preferences)  →  Aligned Assistant
              "here's how to                "here's what humans
               answer nicely"                prefer and don't prefer"
```

**Analogy:** Training a new employee:
1. **SFT** = Show them examples of good work ("this is what we expect")
2. **Reward Model** = Have senior staff rank their outputs ("this response is better than that one")
3. **PPO** = Employee optimizes their work based on the feedback pattern

> **Why RLHF matters:** Without it, LLMs can generate toxic content, follow harmful instructions, or give unhelpful responses. RLHF is what makes the difference between a raw text predictor and a safe, helpful assistant.

---

# Quantization

Quantization reduces a model's **numerical precision** — from 32-bit floating point to 16-bit, 8-bit, or even 4-bit — trading a small accuracy loss for massive savings in memory and speed.

**Analogy:** Instead of measuring ingredients to the microgram (32-bit), you round to the nearest gram (8-bit) or tablespoon (4-bit). For most recipes, the dish tastes the same — but your kitchen needs way less equipment.

| Precision | Memory per param | 7B model size | Quality impact |
|-----------|-----------------|---------------|---------------|
| FP32 (full) | 4 bytes | ~28 GB | Baseline |
| FP16 / BF16 | 2 bytes | ~14 GB | Negligible |
| INT8 | 1 byte | ~7 GB | Very small |
| INT4 | 0.5 bytes | ~3.5 GB | Small but noticeable |

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Load a model in 4-bit quantization — fits on consumer GPUs
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",  # normalized float 4-bit
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
# A 28GB model now fits in ~3.5GB VRAM!
```

> **When to quantize:** When deploying models on consumer hardware, edge devices, or when you need to serve many users with limited GPU memory. Most users interact with quantized models without noticing any quality difference.

---

# RAG (Retrieval-Augmented Generation)

RAG is a pattern that **grounds LLM responses in real, retrieved documents** instead of relying solely on the model's training data. It's the most practical solution to hallucination and knowledge cutoff.

**The pattern:**
1. **Embed** your documents into vectors (using SBERT or similar)
2. **Store** the vectors in a vector database
3. **Retrieve** the most relevant chunks for a given query
4. **Generate** a response using the LLM with the retrieved context included in the prompt

**Analogy:** Instead of asking someone to answer from memory (which may be wrong or outdated), you hand them the relevant pages from a textbook first, then ask them to answer based on those pages.

```python
from sentence_transformers import SentenceTransformer, util
import anthropic

# 1. Your knowledge base (in practice, these are chunked documents)
documents = [
    "Our return policy allows returns within 30 days of purchase with a receipt.",
    "Shipping is free for orders over $50. Standard shipping takes 5-7 business days.",
    "We accept Visa, Mastercard, and PayPal. We do not accept cryptocurrency.",
    "Our customer support hours are Monday-Friday, 9 AM to 5 PM EST."
]

# 2. Embed the documents
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents, convert_to_tensor=True)

# 3. Retrieve relevant documents for a query
query = "Can I return something I bought last week?"
query_embedding = embedder.encode(query, convert_to_tensor=True)
scores = util.cos_sim(query_embedding, doc_embeddings)[0]
top_idx = scores.argmax().item()
context = documents[top_idx]

# 4. Generate answer grounded in the retrieved context
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=[{
        "role": "user",
        "content": f"Based on this context: '{context}'\n\nAnswer: {query}"
    }]
)
print(response.content[0].text)
# "Yes, you can return it. Our policy allows returns within 30 days with a receipt."
```

> **RAG vs Fine-tuning:** RAG gives the model access to current, specific information at query time (no retraining needed). Fine-tuning changes the model's behavior permanently. Use RAG when the knowledge changes frequently (docs, policies). Use fine-tuning when you need a different behavior or style.

## RAG in Production — Chunking, Reranking, and Architecture

The simple RAG example above works for demos. Production RAG systems need careful engineering:

### Document Chunking
Large documents must be split into smaller chunks before embedding. How you chunk dramatically affects retrieval quality.

| Strategy | How it works | Best for |
|----------|-------------|----------|
| **Fixed-size chunks** | Split every N tokens (e.g., 512) with overlap | Simple, general-purpose |
| **Sentence-based** | Split on sentence boundaries | Preserving complete thoughts |
| **Paragraph-based** | Split on paragraph breaks | Well-structured documents |
| **Semantic chunking** | Split when topic changes (using embeddings) | Long documents with topic shifts |
| **Recursive splitting** | Try larger chunks first, split smaller if needed | LangChain default, adaptable |

**Key parameters:**
- **Chunk size:** 256-1024 tokens typical. Smaller = more precise retrieval but less context per chunk. Larger = more context but retrieval noise
- **Chunk overlap:** 10-20% overlap between consecutive chunks to avoid losing information at boundaries

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,     # 100 chars of overlap between chunks
    separators=["\n\n", "\n", ". ", " "]  # try paragraph, then line, then sentence
)

document = "Your long document text here..."
chunks = splitter.split_text(document)
```

### Reranking
Initial retrieval (cosine similarity) is fast but approximate. **Reranking** uses a more powerful model to re-score the top candidates for higher precision.

```
Query → Embedding search (fast, top 20) → Reranker (accurate, picks top 3) → LLM
```

```python
# Using a cross-encoder reranker
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

query = "How do I return a product?"
candidates = ["Return policy...", "Shipping info...", "Refund process..."]

# Score each candidate against the query
scores = reranker.predict([(query, doc) for doc in candidates])
# Reranker reads query + document together → much more accurate than cosine similarity
```

### Hybrid Search
Combine **semantic search** (embeddings) with **keyword search** (BM25/TF-IDF) for the best of both worlds. Semantic search catches meaning; keyword search catches exact terms (product names, error codes).

### Production RAG Architecture

```
User Query
     ↓
[Query Processing]  ← rephrase, expand, extract keywords
     ↓
[Hybrid Retrieval]  ← vector search + keyword search
     ↓
[Reranking]         ← cross-encoder rescoring top candidates
     ↓
[Context Assembly]  ← format retrieved chunks into prompt
     ↓
[LLM Generation]   ← generate answer grounded in context
     ↓
[Citation / Source tracking]  ← link answer back to source documents
```

---

# Vector Databases

Vector databases are purpose-built databases for storing and querying **high-dimensional vectors** (embeddings) by similarity — the backbone of RAG, semantic search, and recommendation systems.

**How they differ from traditional databases:**

| | Traditional DB (SQL) | Vector DB |
|---|---|---|
| **Query type** | Exact match (`WHERE name = 'John'`) | Similarity search ("find most similar to this vector") |
| **Data type** | Rows & columns | High-dimensional vectors |
| **Search method** | Indexes (B-tree, hash) | ANN (Approximate Nearest Neighbor) algorithms |
| **Best for** | Structured queries | Semantic similarity |

**Key players:** ChromaDB, FAISS (Facebook), Pinecone, Weaviate, Qdrant, Milvus

```python
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Create a vector database collection
client = chromadb.Client()
collection = client.create_collection("my_docs")

# 2. Add documents (ChromaDB auto-embeds with its default model)
collection.add(
    documents=[
        "Python is a programming language created by Guido van Rossum.",
        "JavaScript is the language of the web browser.",
        "Rust focuses on memory safety without garbage collection.",
        "Go was designed at Google for concurrent programming."
    ],
    ids=["doc1", "doc2", "doc3", "doc4"]
)

# 3. Query — find most similar documents to your question
results = collection.query(
    query_texts=["What language is good for web development?"],
    n_results=2
)

print(results['documents'])
# [['JavaScript is the language of the web browser.',
#   'Python is a programming language created by Guido van Rossum.']]
```

> **Scaling:** For millions of vectors, use approximate nearest neighbor (ANN) algorithms like HNSW or IVF. They trade perfect accuracy for dramatically faster search — finding the 99.5% best match in milliseconds instead of the 100% best match in seconds.

---

# AI Agents and Tool Use

AI Agents are LLMs that can **take actions in the real world** — not just generate text, but call APIs, run code, search the web, read files, and interact with external systems.

**The shift:** Plain LLMs are like a reference librarian — they answer questions from knowledge. Agents are like a personal assistant — they book flights, send emails, check your calendar, and execute tasks.

## The ReAct Pattern (Reason + Act)

Most agents follow this loop:
1. **Reason** about the task ("I need to check the weather API")
2. **Act** by calling a tool ("call weather_api(city='London')")
3. **Observe** the result ("temperature: 15°C, rain expected")
4. **Reason** again ("The user asked if they need an umbrella — yes, rain is expected")
5. **Respond** with the final answer

## Tool Use / Function Calling

The mechanism that enables agents: define tools (functions) the model can invoke, and the model decides when and how to use them.

```python
import anthropic

client = anthropic.Anthropic()

# Define a tool the model can call
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Do I need an umbrella in London today?"}]
)

# The model decides to call the tool:
for block in response.content:
    if block.type == "tool_use":
        print(f"Model wants to call: {block.name}({block.input})")
        # → Model wants to call: get_weather({"city": "London"})
        # You execute the function, return the result, and the model uses it to answer
```

**Common agent capabilities:**
- **Code execution** — write and run code to solve problems
- **Web search** — look up current information
- **File I/O** — read and write files
- **API calls** — interact with external services
- **Database queries** — look up structured data

### Agent Memory and Planning

Advanced agents maintain **memory across conversations** and **plan multi-step workflows**:

- **Short-term memory:** The conversation context window — what the agent has seen in this session
- **Long-term memory:** Persisted facts stored in databases or files, recalled when relevant (e.g., user preferences, past decisions)
- **Planning:** Agents break complex tasks into sub-tasks, execute them in order, and adapt when steps fail

### Multi-Agent Systems

Instead of one agent doing everything, **multiple specialized agents collaborate**:
- **Orchestrator agent** — routes tasks to specialized sub-agents
- **Research agent** — searches the web, reads documentation
- **Code agent** — writes and executes code
- **Review agent** — checks the output for quality and correctness

> **The future of AI:** Agents represent the shift from "AI that talks" to "AI that does." Claude Code (the tool you're reading this in) is itself an AI agent — it reads your code, searches your codebase, writes files, and runs commands.

---

# Multimodal LLMs

Multimodal models can process and generate **multiple types of content** — text, images, audio, video — in a single model.

**Examples:**
- **GPT-4V / GPT-4o** — text + images (can describe photos, read charts, analyze screenshots)
- **Claude 3.5** — text + images (can interpret diagrams, UI mockups, handwriting)
- **Gemini** — text + images + audio + video
- **Whisper** — audio → text (speech recognition)

**How vision works in practice:**

```python
import anthropic
import base64

client = anthropic.Anthropic()

# Send an image to Claude for analysis
with open("chart.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
            {"type": "text", "text": "Describe what this chart shows. What are the key trends?"}
        ]
    }]
)
print(response.content[0].text)
```

**Real-world multimodal use cases:**
- Analyzing medical images and generating reports
- Processing scanned documents (OCR + understanding)
- Accessibility (describing images for visually impaired users)
- Quality control in manufacturing (inspecting product images)
- Analyzing dashboards and charts from screenshots

---

# LLM Evaluation

How do you measure if an LLM is actually good? This is one of the hardest problems in AI — unlike classification where you have a clear accuracy metric.

## Types of Evaluation

| Approach | How it works | Pros / Cons |
|----------|-------------|-------------|
| **Benchmarks** | Standardized test sets with known answers | Comparable across models, but can be gamed |
| **Human evaluation** | Humans rate quality, helpfulness, safety | Gold standard, but expensive and slow |
| **LLM-as-a-judge** | Use a strong LLM (GPT-4, Claude) to evaluate another LLM's outputs | Scalable, surprisingly good correlation with human ratings |
| **Task-specific metrics** | BLEU (translation), ROUGE (summarization), pass@k (code) | Objective and automated, but narrow |

## Key Benchmarks

| Benchmark | What it tests |
|-----------|-------------|
| **MMLU** | Massive multitask language understanding (57 academic subjects) |
| **HumanEval** | Code generation (can the model write correct Python functions?) |
| **GSM8K** | Grade school math word problems (tests reasoning) |
| **HellaSwag** | Common sense reasoning |
| **MT-Bench** | Multi-turn conversation quality (LLM-as-judge) |
| **LMSYS Chatbot Arena** | Head-to-head human preference voting (most trusted leaderboard) |

**In practice — evaluating YOUR application:**

```python
# Simple LLM-as-judge evaluation for your own use case
import anthropic

client = anthropic.Anthropic()

def evaluate_response(question, response, criteria):
    judge_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Rate this AI response on a scale of 1-5 for: {criteria}

Question: {question}
Response: {response}

Return only a JSON object: {{"score": <1-5>, "reasoning": "<brief explanation>"}}"""
        }]
    )
    return judge_response.content[0].text

# Evaluate helpfulness, accuracy, and safety of your chatbot responses
```

> **Critical insight:** No single metric captures LLM quality. In production, combine automated benchmarks with human spot-checking and real user feedback. Track metrics over time to catch regressions.

---

# Model Deployment and Serving

Training a model is only half the job. **Deploying it into production** so users can access it reliably and efficiently is an entire discipline.

## Common Deployment Patterns

| Pattern | How it works | When to use |
|---------|-------------|-------------|
| **REST API** | Wrap model in a web server, serve predictions via HTTP | Most common. Web apps, mobile apps |
| **Batch inference** | Process large datasets offline on a schedule | Nightly scoring, report generation |
| **Streaming** | Token-by-token generation streamed to the client | Chat interfaces, real-time generation |
| **Edge deployment** | Run model on device (mobile, IoT, browser) | When latency/privacy matters, no internet |

### Simple API Deployment with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model once at startup (not per-request!)
model = torch.load("best_model.pt")
model.eval()

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    with torch.no_grad():
        tensor = torch.tensor([request.features])
        output = model(tensor)
        return PredictionResponse(
            prediction=output.argmax().item(),
            confidence=output.softmax(dim=1).max().item()
        )

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
# Test with: curl -X POST localhost:8000/predict -d '{"features": [1.0, 2.0, 3.0]}'
```

### Production Considerations

| Concern | Solution |
|---------|---------|
| **Latency** | Model optimization (quantization, ONNX, TensorRT), caching, GPU serving |
| **Scalability** | Horizontal scaling, load balancing, auto-scaling (Kubernetes) |
| **Reliability** | Health checks, graceful degradation, fallback models |
| **Monitoring** | Log predictions, track accuracy drift, alert on anomalies |
| **Cost** | Batch similar requests, use smaller models where possible, caching |

### For LLM serving specifically:
- **vLLM** — high-throughput LLM serving with PagedAttention
- **TGI** (Text Generation Inference by HuggingFace) — production LLM server
- **Ollama** — run local LLMs easily on your machine
- **API providers** (Anthropic, OpenAI) — simplest: just call the API

---

# Model Distillation

Model distillation trains a **smaller, faster "student" model** to mimic a large "teacher" model's behavior. The student learns from the teacher's output probabilities (soft labels) rather than the original hard labels.

**Analogy:** Instead of learning physics from a textbook (hard labels), you learn from a brilliant tutor who explains their reasoning and confidence levels for each answer (soft labels). The nuanced explanations help you learn faster and better.

**Why it works:** The teacher model's probability distribution contains rich information. If it predicts "cat: 0.8, tiger: 0.15, dog: 0.04, car: 0.01", the student learns that cats look somewhat like tigers — something binary labels ("cat") don't convey.

**Real-world use:** DistilBERT (the popular distilled version of BERT) retains 97% of BERT's performance at 60% the speed and 40% the size — widely used in production where latency matters.

---

# Responsible AI and Safety

Building AI systems that are **safe, fair, transparent, and beneficial** is not optional — it's a core engineering requirement.

## Key Concerns

| Concern | What it means | Example |
|---------|--------------|---------|
| **Bias** | Model reflects or amplifies biases in training data | Resume screening tool that disadvantages certain demographics |
| **Fairness** | Model performs differently across demographic groups | Facial recognition accuracy varying by skin tone |
| **Hallucination** | LLMs generate false information confidently | Medical chatbot giving incorrect health advice |
| **Privacy** | Models can memorize and leak training data | LLM reproducing personal information from training set |
| **Transparency** | Users should know they're interacting with AI | Chatbot that doesn't disclose it's an AI |
| **Misuse** | AI used for harmful purposes | Deepfakes, automated phishing, misinformation at scale |

## Practical Mitigation

| Strategy | How to implement |
|----------|-----------------|
| **Bias auditing** | Test model performance across demographic subgroups before deployment |
| **Content filtering** | Add safety classifiers to catch harmful inputs and outputs |
| **Human-in-the-loop** | Keep humans in the decision loop for high-stakes applications |
| **Red teaming** | Systematically try to make the model fail or produce harmful output |
| **Guardrails** | Set boundaries on what the model can/cannot do (system prompts, output validation) |
| **Transparency** | Document model capabilities, limitations, and training data |

> **The industry reality:** Every major AI company now has responsible AI teams and policies. Regulations like the EU AI Act are making safety practices legally required, not just ethically important. As an AI practitioner, understanding these concerns is as essential as understanding backpropagation.
