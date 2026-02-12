# 🔍 Semantic Text Compression Using NLP

## 📌 Overview

This project implements a hybrid text compression system that combines **Sentence-BERT (SBERT)** for semantic redundancy removal with **Huffman Coding** and **zlib (DEFLATE)** for lossless compression.

Unlike traditional compression methods that only reduce character-level redundancy, this system removes both **semantic and statistical redundancy**, achieving higher compression while preserving the original meaning of the text.

---

## 🚀 Features

* Semantic compression using SBERT
* Cosine similarity-based redundant sentence removal
* Huffman Coding for entropy-based compression
* zlib for additional lossless compression
* Interactive Streamlit web interface
* Real-time compression ratio comparison

---

## 🧠 Tech Stack

Python, SBERT, Transformers, NLTK, Huffman Coding, zlib, Streamlit

---

## ⚙️ How It Works

1. Text input and preprocessing
2. Semantic similarity detection using SBERT
3. Removal of redundant sentences
4. Lossless compression using Huffman Coding and zlib
5. Compression ratio evaluation

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Results

* 15–30% reduction using semantic compression
* Up to 45–60% overall compression using the hybrid model
* Preserves over 90% semantic similarity with the original text

---

## 📌 Applications

Efficient document storage, bandwidth optimization, intelligent text compression, and crisis communication systems.
