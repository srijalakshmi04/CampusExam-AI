# 🎓 CampusExam AI

### AI-Powered College FAQ Chatbot for Examination Queries

<p align="center">
  <b>Offline NLP-Based College Examination Helpdesk & FAQ Assistant</b>
</p>

---

## 📌 Project Overview

**CampusExam AI** is a College FAQ Chatbot developed to answer frequently asked **examination-related queries** using Natural Language Processing (NLP).

The chatbot allows students to ask questions in natural language instead of searching through documents or notices manually.

It uses:

- **TF-IDF Vectorization**
- **Cosine Similarity**
- **32 FAQ intents**
- **206 training patterns**
- **Offline NLP processing**

The system identifies the FAQ that is most similar to the student's question and provides the corresponding answer.

---

## 🎯 Objective

The main objective of this project is to develop a simple and interactive College FAQ Chatbot that can provide quick answers to common examination-related questions.

The chatbot is designed to help students with queries related to:

- Semester examinations
- Hall tickets
- Attendance requirements
- Examination rules
- Results
- Revaluation
- Retotalling
- Supplementary examinations
- Calculators and stationery
- Examination absence
- Internal examinations
- Examination registration

---

## 🤖 NLP Approach

The chatbot uses a similarity-based NLP approach.

### 1. TF-IDF Vectorization

TF-IDF (**Term Frequency-Inverse Document Frequency**) converts text into numerical vectors based on the importance of words.

This allows the system to represent both:

- Student's question
- Stored FAQ questions

as numerical vectors.

### 2. Cosine Similarity

Cosine Similarity measures how similar two text vectors are.

The chatbot compares the user's question with the available FAQ patterns and selects the most relevant matching intent.

### Basic Workflow

```text
Student Question
       ↓
Text Preprocessing
       ↓
TF-IDF Vectorization
       ↓
Cosine Similarity
       ↓
Find Most Similar FAQ
       ↓
Identify Intent
       ↓
Generate Answer





