# 🏠 ML-Based Buyer Segmentation and Investment Profiling
## Real Estate Market Intelligence — Parcl.co

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![ML](https://img.shields.io/badge/ML-K--Means%20Clustering-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 🌐 Live Dashboard
### 👉 [Click Here to Open Live Dashboard](https://parcl-buyer-segmentation.streamlit.app/)

---

## 🏢 Internship Details
| Field | Details |
|-------|---------|
| **Organization** | Unified Mentor |
| **Program** | Data Analytics Internship (6-Month) |
| **Project Number** | Project 1 of 5 |
| **Submitted By** | Mohan |
| **Date** | May 2026 |

---

## 📋 Project Overview
AI-driven buyer segmentation system for Parcl real estate platform.
Applies K-Means and Hierarchical Clustering to 2,000 buyers and
10,000 property transactions to identify 4 distinct buyer segments
for targeted marketing and investment profiling.

---

## 🎯 4 Buyer Segments Discovered

| Segment | Buyers | Share | Avg Price | Key Feature |
|---------|--------|-------|-----------|-------------|
| 🔵 C1 - Global Investors | 851 | 42.6% | $391,299 | Highest spend, most properties |
| 🟢 C2 - First-Time Buyers | 1,001 | 50.1% | $309,015 | Largest group, highest invest intent |
| 🟠 C3 - Corporate Buyers | 53 | 2.6% | $347,987 | France-based, highest loan rate |
| 🟣 C4 - Luxury Investors | 95 | 4.8% | $351,759 | UK-based, premium segment |

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| pandas, numpy | Data manipulation |
| scikit-learn | K-Means and Hierarchical Clustering |
| matplotlib, seaborn | Static visualizations (10 charts) |
| Plotly | Interactive dashboard charts |
| Streamlit | Web dashboard deployment |
| Jupyter Notebook | Development environment |
| VS Code | Code editor |

---

## 📁 Project Files
| File | Description |
|------|-------------|
| `buyer_segmentation.ipynb` | Complete ML pipeline (10 phases) |
| `app.py` | Interactive Streamlit dashboard |
| `clients.csv` | Raw client dataset (2,000 buyers) |
| `properties.csv` | Raw property dataset (10,000 listings) |
| `buyers_segmented.csv` | Final clustered output with segment labels |
| `requirements.txt` | Python dependencies for deployment |

---

## 🔬 Methodology
1. Data Cleaning (sale_price, date_of_birth, duplicates)
2. Feature Engineering (merge datasets, derive 6 new features)
3. Feature Encoding (Label + One-Hot Encoding)
4. Feature Scaling (StandardScaler)
5. Optimal K Selection (Elbow Method + Silhouette Score)
6. K-Means Clustering (K=4)
7. Hierarchical Clustering Validation (ARI Score)
8. EDA Visualization (10 professional charts)
9. Streamlit Dashboard Deployment

---

## 📊 Key Results
- ✅ 2,000 buyers segmented into 4 distinct groups
- ✅ K=4 validated by Elbow Method + Silhouette
