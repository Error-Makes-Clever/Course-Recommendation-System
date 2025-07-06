# 🎓 Personalized Course Recommendation System

A smart, AI-powered course recommender built using hybrid collaborative filtering, machine learning, and deep learning models — deployed seamlessly with Supabase and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?logo=python)
![Framework](https://img.shields.io/badge/Framework-Streamlit-green)
<a href="">→Live Demo<a/> 
---

## 📌 Overview

This project recommends personalized learning courses to users based on their learning history using multiple algorithms. It supports both **new** and **existing users**, offering tailored recommendations through multiple ML/Deep Learning techniques:

- ✅ Collaborative Filtering
- ✅ Clustering with KMeans + PCA
- ✅ Neural Collaborative Filtering (NCF)
- ✅ Embedding-based Regression & Classification
- ✅ Content-based Course Similarity
- ✅ User Profile Modeling

All models are trainable and deployable directly from the Streamlit interface, integrated with Supabase for data persistence.

---

## 🧠 Features

- 📈 **Dynamic Model Selection**: Choose from various algorithms based on the user’s data.
- 🧩 **NCF (Neural Collaborative Filtering)**: For deep learning-based recommendations.
- 🧠 **Clustering with PCA**: Groups similar users for smarter predictions.
- 📚 **Content-Based Filtering**: Uses course metadata and BOW representations.
- 🧠 **Regression & Classification Embedding Models**: Generate custom user/item features for improved accuracy.
- ☁ **Supabase Integration**: Fully managed backend for storing models, ratings, and user data.
- 📊 **EDA, KMeans Elbow, Similarity Heatmaps, Hyperparameter Tuning**: Visualized and saved as part of analysis.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **Pandas, NumPy** | Data manipulation |
| **scikit-learn** | ML algorithms & preprocessing |
| **TensorFlow/Keras** | Deep learning models |
| **Supabase** | Backend DB and Storage |
| **Streamlit** | Web app interface |
| **AgGrid** | Interactive tables |
| **dotenv** | Secure env handling |

---

## 🚀 How It Works

1. **User selects or enters their ID**
2. **Model training**:
   - Streamlit triggers model training via backend logic
   - Models saved in Supabase bucket as `.xz` compressed pickle files
3. **Prediction phase**:
   - Trained models are loaded from storage
   - Personalized recommendations generated and displayed with confidence scores

---

## 📂 Project Structure

```plaintext
📁 backend/
  ├── recommendation_models.py         # Core recommendation logic
  ├── retrain_NCF.py                   # GitHub Actions: retrain script
  └── supabase_utils.py                # Supabase interaction layer

📁 .github/workflows/
  └── retrain_ncf.yml                  # Auto-training with GitHub Actions (Daily 6:30 PM IST)

📁 frontend/
  └── streamlit_app.py                 # UI interface built with Streamlit

📁 assets/
  ├── eda.png                          # Exploratory Data Analysis
  ├── course_similarity_heatmap.png   # Cosine similarity between courses
  ├── kmeans_elbow.png                # Optimal clusters in user data
  ├── regression_tuning.png           # Hyperparameter tuning (Regression)
  ├── classification_tuning.png       # Hyperparameter tuning (Classification)

📄 requirements.txt
📄 .env.example
📄 README.md
```
## 📸 Screenshots

| EDA & Similarity Heatmap | KMeans Elbow | Model Results |
|--------------------------|--------------|----------------|
| ![](assets/eda.png) | ![](assets/kmeans_elbow.png) | ![](assets/regression_tuning.png) |

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/course-recommendation-system.git
cd course-recommendation-system
```
### 2. Create and activate virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
 ```
### 3. Install dependencies
```
pip install -r requirements.txt

```
### 4. Set up environment variables
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```
### 5. Run Streamlit app
```
streamlit run frontend/streamlit_app.py

```

## 📢 Future Improvements
📌 Real-time feedback integration from users

📊 Model evaluation dashboard

🤖 Model Explainability (SHAP, LIME)

🌐 Deploy as a full-stack SaaS product

## ⭐ Support
If you found this project helpful, consider giving it a ⭐ star on GitHub!
