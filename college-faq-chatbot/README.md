# 🎓 CampusExam AI: Development of a College FAQ Chatbot Using Natural Language Processing

A practical Artificial Intelligence & NLP project designed to assist college students with all inquiries regarding **examinations, hall tickets, timetable schedules, attendance regulations, revaluation, and academic rules**.

---

## 📌 1. Project Title & Overview
- **Project Title:** Development of a College FAQ Chatbot Using Natural Language Processing
- **Domain / Category:** College Examinations
- **Chatbot Name:** CampusExam AI
- **Course / Level:** Practical Artificial Intelligence / B.Tech AI & Data Science
- **Deployment Mode:** 100% Local & Offline (No paid cloud APIs or API keys required)

---

## 🎯 2. Problem Statement & Objectives

### Problem Statement
College examination cells receive hundreds of repetitive student queries every semester regarding exam timetables, admit cards, fee deadlines, attendance condonation, and revaluation rules. Manual support often leads to delays, long queues, and miscommunication.

### Objectives
1. Build an automated, lightweight, and explainable **Natural Language Processing (NLP)** chatbot.
2. Enable students to ask questions in free-form English (e.g. *"when is my sem exam?"*, *"where to get admit card?"*) without needing rigid keywords.
3. Utilize **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to compute semantic relevance.
4. Provide immediate, accurate, and clearly structured guidance with confidence metrics and interactive topic suggestions.

---

## 🧠 3. AI / NLP Architecture & How It Works

CampusExam AI follows a structured, transparent 6-stage NLP pipeline:

```text
[ User Question ]
       │
       ▼
1. Text Normalization & Preprocessing
   ├── Lowercasing ("When Are Sem Exams?" -> "when are sem exams?")
   ├── Abbreviation/Synonym Expansion ("sem" -> "semester", "hallticket" -> "hall ticket")
   ├── Special Character Stripping (removes punctuation, symbols)
   └── Tokenization & Stop Words Removal
       │
       ▼
2. Feature Extraction (N-Grams & TF-IDF Vectorization)
   ├── Unigram + Bigram generation: ["semester", "exams", "semester exams"]
   ├── Term Frequency (TF): Measures word occurrence inside the query
   └── Inverse Document Frequency (IDF): Boosts distinctive domain terms
       │
       ▼
3. Vector Space Modeling (L2 Normalization)
   └── Represents user query and 100+ FAQ patterns as high-dimensional numerical vectors
       │
       ▼
4. Cosine Similarity Measurement
   └── Computes dot product: CosineSim(u, v) = (u · v) / (||u|| * ||v||)
       │
       ▼
5. Intent Classification & Confidence Thresholding
   ├── If Cosine Similarity >= 0.22 ──> Return Matched FAQ Intent & Verified Answer
   └── If Cosine Similarity < 0.22  ──> Trigger Graceful Fallback with Topic Suggestions
       │
       ▼
[ Chatbot Response with Confidence Score & Topic Chips ]
```

### Mathematical Formulation
1. **Term Frequency ($TF$):**
   $$TF(t, d) = 1 + \log(f_{t, d}) \quad \text{for } f_{t, d} > 0$$

2. **Inverse Document Frequency ($IDF$):**
   $$IDF(t) = \log\left(\frac{1 + N}{1 + DF(t)}\right) + 1$$
   *Where $N$ is total FAQ corpus patterns, and $DF(t)$ is document frequency of term $t$.*

3. **Cosine Similarity ($CosSim$):**
   $$\text{Cosine Similarity}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2}$$

---

## 🗂️ 4. Project Structure

```text
college-faq-chatbot/
│
├── app.py                     # Flask application server & NLP matching engine
├── requirements.txt           # Python dependencies (Flask, scikit-learn, numpy)
├── README.md                  # Comprehensive academic documentation & viva guide
├── data/
│   └── faq.json               # 32 Examination Intents & 100+ training patterns
├── templates/
│   └── index.html             # Modern, responsive academic web interface
├── static/
│   ├── style.css              # Custom styling, animations & theme design
│   └── script.js              # Asynchronous AJAX chat client & NLP inspector
└── screenshots/               # Application UI and demonstration captures
```

---

## 🛠️ 5. Technology Stack

| Layer | Technology | Details |
| :--- | :--- | :--- |
| **Frontend** | HTML5, Vanilla CSS3, JavaScript (ES6) | Responsive layout, modern UI, modal dialogs, typing indicator |
| **Backend** | Python Flask 3.x | Lightweight RESTful microframework |
| **NLP Engine** | `scikit-learn`, `numpy`, Custom NLP Module | TF-IDF Vectorizer, Cosine Similarity, Regex Preprocessing |
| **Knowledge Base**| JSON | 32 examination categories & intents |

---

## ⚡ 6. Installation & Execution Guide

### Prerequisites
- Python 3.9+ or Python 3.11+
- `pip` package manager

### Step 1: Clone or Navigate to Project Directory
```powershell
cd C:\Users\Lenovo\.gemini\antigravity-ide\scratch\college-faq-chatbot
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Flask Application
```powershell
python app.py
```

### Step 4: Open in Web Browser
Navigate to:
```text
http://127.0.0.1:5000
```

---

## 💬 7. Demonstration Test Cases (8 Core Evaluation Questions)

You can demonstrate these exact questions to professors or evaluators:

| # | User Question | Predicted Intent | Expected Chatbot Response |
| :- | :--- | :--- | :--- |
| 1 | *"When are the semester exams?"* | `exam_schedule` | Explains academic calendar schedule (Odd: Nov/Dec, Even: Apr/May). |
| 2 | *"Where can I get my hall ticket?"* | `hall_ticket` | Details portal download & department distribution guidelines. |
| 3 | *"What is the attendance requirement?"* | `attendance` | Details mandatory 75% rule and medical condonation (65%-74%). |
| 4 | *"How can I apply for revaluation?"* | `revaluation` | Details 7-14 day revaluation window, portal form, and fees. |
| 5 | *"When will the results be published?"* | `results` | Details standard 4-6 week publication timeline and portal link. |
| 6 | *"What should I bring to the exam hall?"* | `required_stationery`| Details hall ticket, ID card, pens, and transparent pouch rules. |
| 7 | *"Can I use a calculator?"* | `calculator_rules` | Explains non-programmable scientific calculator permissions. |
| 8 | *"What happens if I miss an exam?"* | `absence` | Explains 'Absent' arrear status and supplementary exam policy. |

---

## 📊 8. Examination FAQ Coverage (32 Topics)

The dataset in `data/faq.json` covers 32 examination topics:
1. `exam_schedule` - Semester examination dates & commencement
2. `internal_exam` - Mid-term tests & Continuous Assessment (CIA)
3. `end_sem_exam` - End semester comprehensive theory papers
4. `exam_timetable` - Timetable schedule & session timings (FN/AN)
5. `exam_hall` - Exam room & seating allocation notice boards
6. `hall_ticket` - Hall ticket / Admit card issuance & rules
7. `exam_registration` - Semester exam registration & form submission
8. `exam_fee` - Examination fee structure & payment deadlines
9. `attendance` - 75% attendance threshold & shortage criteria
10. `eligibility` - Prerequisites for appearing in semester exams
11. `supplementary_exam` - Remedial / Backlog examination procedure
12. `revaluation` - Answer sheet re-evaluation application & fee
13. `retotalling` - Mark summation verification & paper photocopy
14. `results` - Result declaration date & online mark statements
15. `internal_marks` - Continuous internal assessment mark calculation
16. `passing_criteria` - Minimum pass percentages (theory & practical)
17. `exam_duration` - Paper duration (3 hours vs 2 hours)
18. `reporting_time` - Entry time (30 mins before) & late entry restrictions
19. `required_stationery` - Permitted pens, geometry tools, and pouches
20. `calculator_rules` - Non-programmable scientific calculator policy
21. `mobile_restrictions` - Strict ban on mobile phones & smartwatches
22. `dress_id_requirements` - Mandatory college ID card & formal dress code
23. `absence` - Absenteeism consequences & arrear registration
24. `medical_leave` - Hospitalization & medical condonation protocol
25. `arrears` - Backlogs clearing timeframe & university limits
26. `practical_exam` - Lab practical examinations, external examiner & records
27. `project_viva` - Final year project demonstration & viva-voce
28. `malpractice` - Disciplinary actions, malpractice committee & penalties
29. `scribe_assistance` - Scribe facility & compensatory time for PwD
30. `exam_cell_contact` - Office of Controller of Examinations (CoE)
31. `grade_improvement` - Betterment exam rules for passed papers
32. `duplicate_docs` - Duplicate hall tickets & official transcripts

---

## 🎓 9. Viva Voce Questions & Answers (For Oral Examination)

#### Q1: What is the main objective of this project?
> **Ans:** To develop a lightweight, offline Natural Language Processing chatbot that answers student inquiries about college examinations using semantic text matching rather than rigid keyword searches.

#### Q2: Why use TF-IDF and Cosine Similarity instead of simple if-else matching?
> **Ans:** Keyword matching fails when students use synonyms, typos, or varied sentence structures (e.g. *"when are sem exams?"* vs *"exam date sheet"*). TF-IDF and Cosine Similarity calculate semantic geometric proximity in high-dimensional vector space.

#### Q3: What happens during the Text Preprocessing step?
> **Ans:** The raw input text undergoes lowercasing, punctuation stripping, domain abbreviation expansion (e.g. *sem* &rarr; *semester*, *admit card* &rarr; *hall ticket*), tokenization, and stop word removal.

#### Q4: What is the purpose of IDF (Inverse Document Frequency)?
> **Ans:** IDF downweights words that appear everywhere across all documents (like *"exam"*, *"college"*) and gives higher mathematical significance to rare, discriminative words (like *"revaluation"*, *"calculator"*, *"scribe"*).

#### Q5: What is Cosine Similarity and what is its range?
> **Ans:** Cosine similarity measures the cosine of the angle between two non-zero vectors. Its value ranges from `0.0` (completely orthogonal/independent) to `1.0` (identical orientation).

#### Q6: How does the system handle unknown or nonsensical questions?
> **Ans:** We enforce a confidence threshold (0.22). If no FAQ pattern achieves a similarity score above 0.22, the system triggers a fallback message prompting the student with supported topic options.

#### Q7: Why are N-grams (1, 2) included in the vectorizer?
> **Ans:** Bigrams capture crucial two-word phrases such as *"hall ticket"*, *"internal exam"*, and *"passing criteria"*, preserving essential contextual meaning that unigrams alone might lose.

#### Q8: Does this application require an active internet connection or paid cloud APIs?
> **Ans:** No. The entire application runs 100% locally and offline using Python Flask and Scikit-Learn.

#### Q9: How is the knowledge base structured?
> **Ans:** The knowledge base is stored in a clean JSON format containing intents, categories, training phrase variations, verified answers, and interactive quick-reply chips.

#### Q10: What are the current limitations and future scope?
> **Ans:** Current limitations include single-turn question-answering. Future scope includes multi-turn dialogue memory, Transformer-based embeddings (e.g., MiniLM/BERT), voice input support, and integration with the college ERP database.

---

## 🔒 10. Error Handling & Edge Cases
- **Empty Queries:** Validated on both client and server sides.
- **Out-of-domain / Low Confidence:** Handled gracefully via threshold fallback.
- **Server Disconnections:** Catches network drops and renders user-friendly offline alerts.
- **Special Characters / Malformed JSON:** Sanitized safely without crashing Flask.

---

## 📜 11. Academic Disclaimer
*This chatbot provides generalized university examination guidance for academic demonstration purposes. Official dates, fee amounts, and regulations should always be verified through official college examination notices.*
