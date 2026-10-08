# 🎓 Student Placement AI Predictor
> **Machine Learning Platform for Career Trajectory Forecasting, Diagnostic Skill Analytics, and Cohort Evaluation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Theme](https://img.shields.io/badge/UI%2FUX-Monochrome%20Pro-black.svg)]()

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement & Target Audience](#-problem-statement--target-audience)
- [Key Features & Capabilities](#-key-features--capabilities)
- [Machine Learning Architecture & Algorithms](#-machine-learning-architecture--algorithms)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Application Modules](#-application-modules)
- [Technology Stack & Libraries Breakdown](#️-technology-stack--libraries-breakdown)
- [Project Directory Structure](#-project-directory-structure)
- [Local Installation & Setup](#-local-installation--setup)
- [Batch Processing Workflow](#-batch-processing-workflow)
- [How to Host It Live Online for Free](#-how-to-host-it-live-online-for-free)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 📖 Project Overview

The **Student Placement AI Predictor** is an interactive, production-ready machine learning application designed to forecast student campus placement outcomes based on academic history, technical aptitude, cognitive skills, and practical project experience. 

Built using **Random Forest Classification**, **Plotly**, and **Streamlit**, the platform delivers real-time placement probability scoring, multi-dimensional peer benchmarking, counterfactual "what-if" simulations, and high-throughput batch evaluation for entire student cohorts.

Designed with an ultra-clean, mature **Monochrome (Black & White)** aesthetic inspired by high-end developer interfaces (Vercel, Linear, Apple), the UI prioritizes clarity, data density, and zero-distraction analytics.

---

## 🎯 Problem Statement & Target Audience

### The Problem
During collegiate campus placement cycles, students often discover deficiencies in their academic and technical profiles too late in the process. Concurrently, **Training & Placement Offices (TPOs)** lack automated, data-driven systems to audit student readiness at scale, leading to sub-optimal placement conversion rates and misallocated remedial training.

### Who Uses This Platform?
1. **Students & Job Aspirants**:
   - Identify placement viability months ahead of recruitment drives.
   - Discover specific skill vulnerabilities (e.g., active backlogs, insufficient coding score).
   - Simulate counterfactual improvements (e.g., *"What happens if I complete 1 more internship and clear my backlog?"*).
2. **College Placement Offices (TPOs) & Academic Deans**:
   - Ingest entire student batches (hundreds of profiles) via CSV.
   - Automatically triage cohorts into "High Confidence" and "At-Risk" segments.
   - Direct mentorship, mock interviews, and training resources to the students who need them most.
3. **Career Mentors & Counselors**:
   - Provide concrete, evidence-backed advice grounded in historical empirical recruitment data.

---

## ✨ Key Features & Capabilities

- **Instant Placement Probability Engine**: Estimates placement odds with calibrated probabilities and confidence levels (*High Confidence*, *Moderate Confidence*, *Elevated Risk*, *Critical Risk*).
- **Quick-Load Preset Archetypes**: 1-click candidate presets (*Top Scholar*, *Tech Specialist*, *Borderline Profile*, *High-Risk Candidate*) for instant scenario exploration.
- **Dynamic Diagnostic Signals**: Automatically isolates positive recruitment catalysts (`+`) versus profile vulnerabilities (`-`).
- **Strategic Milestone Guidance**: Concrete, actionable guidance customized to the candidate's active standing.
- **Multi-Skill Radar Benchmark**: Compares student metrics against the empirical mean of placed students across 6 normalized axes.
- **What-If Sensitivity Sandbox**: Interactive sliders to simulate the impact of hypothetical interventions (e.g., boosting coding score by +15, clearing all backlogs).
- **High-Throughput Batch Processing**: Evaluates rosters of students from CSV files with aggregate KPI metrics, ratio donuts, probability distribution histograms, and downloadable prediction rosters.
- **Audited Diagnostics Dashboard**: Confusion matrix heatmap, ROC curve with AUC analysis, Gini feature importance ranking, and full classification report.
- **Dataset Explorer & Filtering**: Interactive bivariate scatter plots, CGPA distribution box plots, correlation heatmaps, and customizable CSV exports.

---

## 🧠 Machine Learning Architecture & Algorithms

### 1. Classification Algorithm: Random Forest Classifier
The underlying model uses **Random Forest Classification** (`sklearn.ensemble.RandomForestClassifier`), an ensemble bagging technique that constructs a multitude of decision trees during training.

#### Why Random Forest for Placement Prediction?
1. **Handles Non-Linear Thresholds**: Campus hiring relies on non-linear rules (e.g., having active backlogs often disqualifies candidates regardless of high aptitude; conversely, exceptional coding ability can offset an average CGPA). Decision trees naturally model these non-linear thresholds without requiring artificial polynomial transformations.
2. **Mitigates Overfitting (Bagging & Subsampling)**: By aggregating predictions across 180 decorrelated decision trees, Random Forest significantly reduces variance compared to single decision trees or logistic regressors.
3. **Scale Invariance**: Tree-based partition splits are invariant to monotonic transformations, allowing features with disparate scales (e.g., CGPA $0-10$ vs. Attendance $0-100\%$) to interact harmoniously without distortion.
4. **Interpretability via Gini Impurity**: Provides exact feature importance metrics, showing which factors most heavily dictate recruitment decisions.

### 2. Model Hyperparameters & Configuration
```python
RandomForestClassifier(
    n_estimators=180,      # Number of decision trees in the ensemble
    max_depth=8,           # Constrains tree depth to prevent overfitting/memorization
    min_samples_leaf=2,    # Ensures terminal leaves represent generalized sub-samples
    random_state=42        # Enforces deterministic, fully reproducible results
)
```

### 3. Feature Space (9 Input Predictors)

| Feature | Type | Range | Description & Role |
| :--- | :--- | :--- | :--- |
| **CGPA** | `float` | `0.00 – 10.00` | Cumulative Grade Point Average; foundational eligibility filter. |
| **Coding_Score** | `int` | `0 – 100` | Hands-on programming and data structures benchmark. |
| **Aptitude_Score** | `int` | `0 – 100` | Quantitative, logical, and verbal reasoning screening score. |
| **Communication_Score** | `int` | `0 – 100` | Soft skills, interpersonal fluency, and behavioral assessment. |
| **Internships** | `int` | `0 – 10` | Real-world industry internship experiences completed. |
| **Projects** | `int` | `0 – 10` | Academic and open-source software projects built. |
| **Attendance** | `int` | `0 – 100%` | Classroom attendance rate; indicator of discipline. |
| **Certifications** | `int` | `0 – 10` | Industry-verified technical skill certifications. |
| **Backlogs** | `int` | `0 – 10` | Standing active uncleared academic backlogs. |

**Target Variable (`Placed`)**: Binary classification (`1`: Placed, `0`: Not Placed).

---

## 📊 Model Performance & Evaluation

The model is trained on an **80% stratified training partition (480 records)** and evaluated on the **20% held-out test partition (120 records)**:

```
                      Precision    Recall    F1-Score    Support
----------------------------------------------------------------
Not Placed (Class 0)    0.881       0.952      0.915       62
Placed     (Class 1)    0.943       0.862      0.901       58
----------------------------------------------------------------
Overall Accuracy:       90.83%
ROC-AUC Metric:         0.972
```

### Performance Summary:
- **Test Accuracy (90.83%)**: 109 out of 120 test candidates are classified correctly.
- **Precision (94.34%)**: When the model predicts a candidate will be placed, it is accurate **94.34%** of the time, minimizing false positives.
- **ROC-AUC (0.972)**: Indicates near-optimal separability across varying decision thresholds.

---

## 🖥️ Application Modules

### 1. `Predict & Simulate`
- Interactive form with grouped inputs for academic standing, technical competence, and practical exposure.
- Real-time gauge speedometer displaying calibrated placement probability.
- Automated list of strengths and vulnerabilities.
- Multi-dimensional spider/radar chart overlaying candidate against placed peers.
- What-If Sensitivity Simulator to test hypothetical improvements.

### 2. `Batch Evaluation`
- Bulk evaluation of entire classes or cohorts via CSV upload.
- One-click testing using a pre-packaged 25-student benchmark roster.
- Aggregate KPI cards (Total Candidates, Projected Placements, At-Risk Count, Average Probability).
- Donut outcome charts and probability distribution histograms.
- Full prediction table with individual probabilities and single-click CSV export.

### 3. `Model Performance`
- Live metrics overview (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`).
- Interactive Confusion Matrix (Sapphire-to-Cyan heatmap) with true negative/positive counts.
- High-resolution ROC Curve with shaded area under curve.
- Gini Feature Importance horizontal chart.
- Complete classification report and model specifications.

### 4. `Dataset Explorer`
- High-level empirical distribution counters.
- Bivariate CGPA distribution box plots.
- Coding vs. Aptitude scatter clusters with project sizing.
- Pairwise feature correlation matrix heatmap.
- Filterable data browser with custom CSV export.

---

## 🛠️ Technology Stack & Libraries Breakdown

| Category | Technology / Library | Version | Role in the Platform |
| :--- | :--- | :--- | :--- |
| **Language** | **Python** | `3.10+` | Core programming language powering ML logic and data transformations. |
| **Web Framework** | **Streamlit** | `>=1.40` | Reactive dashboard architecture, session state management, caching (`@st.cache_resource`, `@st.cache_data`), file uploaders, and CSV downloads. |
| **Machine Learning** | **Scikit-Learn** | `>=1.4` | Implementation of `RandomForestClassifier`, stratified 80/20 train/test splitting, confusion matrix, ROC curve, AUC score, and classification reports. |
| **Data Manipulation** | **Pandas** | `>=2.0` | Tabular data structures (`DataFrame`), multi-column feature selection, batch CSV parsing, statistical aggregations, and CSV exports. |
| **Numerical Computing** | **NumPy** | `>=1.24` | Vectorized mathematical operations, Poisson/Gaussian random distributions, probability rounding, and array transformations. |
| **Interactive Visuals** | **Plotly** | `>=5.18` | Hardware-accelerated interactive web charts: placement gauge meters, radar spider benchmarks, confusion matrix heatmaps, ROC curves, and cohort histograms. |
| **Scientific Charting** | **Matplotlib** | `>=3.7` | Static plotting utilities and complementary visualization primitives. |
| **Statistical Visuals** | **Seaborn** | `>=0.13` | Statistical data exploration and correlation matrix styling. |
| **Deployment** | **Streamlit Cloud** | — | Cloud hosting platform with automated continuous deployment on `git push`. |

---

## 📁 Project Directory Structure

```plaintext
student-placement-ml/
├── .streamlit/
│   └── config.toml                  # Streamlit dark monochrome theme & server config
├── .gitignore                       # Git exclusion rules (.venv, caches, etc.)
├── app.py                           # Main Streamlit application and ML pipeline
├── generate_data.py                 # Synthetic dataset generation script
├── requirements.txt                 # Project dependencies
├── student_placement.csv            # Empirical student placement dataset (600 rows)
├── Practical_10_Student_Placement_Report.docx # Academic lab report
└── README.md                        # Documentation
```

---

## 💻 Local Installation & Setup

### Prerequisites
- **Python 3.10+** installed on your system.
- Git (optional, for cloning).

### 1. Clone or Open the Workspace
```bash
git clone https://github.com/<your-username>/student-placement-ml.git
cd student-placement-ml
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to access the dashboard.

---

## 👥 Batch Processing Workflow

To evaluate multiple students at once:

1. Navigate to the **Batch Evaluation** page from the sidebar.
2. Download the pre-formatted template by clicking **"Download Benchmark Template (25 Students)"**.
3. Format your student data CSV with the following 9 exact column headers:
   ```csv
   CGPA,Aptitude_Score,Coding_Score,Communication_Score,Internships,Projects,Attendance,Certifications,Backlogs
   8.87,94,79,77,3,2,76,0,0
   7.19,63,69,52,2,3,71,1,1
   9.29,86,87,94,3,1,93,1,0
   ```
4. Upload your CSV into the file uploader.
5. The platform will instantaneously generate summary statistics, cohort charts, individual predictions, and provide an **"Export Evaluated Cohort Data (CSV)"** download button.

---

## 🌐 How to Host It Live Online for Free

You can deploy this application online in less than 3 minutes so anyone can access it via a public URL:

### Method 1: Streamlit Community Cloud (Recommended — 100% Free)

Streamlit Community Cloud is the fastest, free, and official hosting platform for Streamlit applications.

#### Step 1: Initialize Git and Push to GitHub
Open your terminal in the project directory and run:

```bash
# Initialize git repository
git init

# Add all files (the .gitignore ensures .venv is excluded)
git add .

# Create initial commit
git commit -m "Initial commit: Student Placement AI Predictor"

# Link to your GitHub repository
git branch -M main
git remote add origin https://github.com/<your-username>/student-placement-ml.git
git push -u origin main
```

#### Step 2: Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<your-username>/student-placement-ml`.
4. Set the **Branch** to `main`.
5. Set the **Main file path** to `app.py`.
6. Click **"Deploy!"**.

Streamlit Cloud will read `requirements.txt`, install dependencies, load `.streamlit/config.toml`, and publish your app with a public URL (e.g. `https://student-placement-ai.streamlit.app`).

---

### Method 2: Hugging Face Spaces (Alternative Free Host)

1. Create a free account at **[huggingface.co](https://huggingface.co)**.
2. Go to **Spaces** $\rightarrow$ Click **"Create new Space"**.
3. Enter a Space Name (e.g., `student-placement-predictor`).
4. Select **Streamlit** as the Space SDK.
5. Choose **Public** and select the free CPU hardware tier.
6. Push your repository code to the Hugging Face Git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/student-placement-predictor
   git push space main
   ```
7. Hugging Face will build and launch your live application automatically!

---

### Method 3: Render (Free Web Service)

1. Sign up at **[render.com](https://render.com)**.
2. Click **"New"** $\rightarrow$ **"Web Service"** and connect your GitHub repository.
3. Configure the service settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **"Create Web Service"**.

---

## 📜 License & Citation

This project is licensed under the **MIT License**. It was developed as an educational and operational machine learning framework for academic evaluation and campus placement analytics.

If you use or adapt this codebase in your academic projects, please cite:
```bibtex
@misc{studentplacementml2026,
  title={Student Placement AI Predictor: High-Precision Ensemble Classification and Diagnostic Analytics},
  author={Practical 10 ML Team},
  year={2026}
}
```
