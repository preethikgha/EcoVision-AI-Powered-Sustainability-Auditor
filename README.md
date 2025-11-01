#  EcoVision: AI-Powered Sustainability Auditor

> *“AI that sees sustainability — transforming how brands and consumers understand eco-friendliness.”*

---

##  Overview

**EcoVision** is a **Computer Vision + Data Science** project that uses **deep learning** to classify product packaging as **eco-friendly** or **non-sustainable**.  
The system combines **image classification, explainable AI (Grad-CAM)**, and **data analytics dashboards** to promote **green retail innovation** and **sustainability awareness**.

This project aligns with **UN Sustainable Development Goals (SDG 12 & SDG 13)** — focusing on *Responsible Consumption* and *Climate Action* — by leveraging AI to audit packaging sustainability in the retail industry.

---

##  Problem Statement

Today’s retail industry lacks scalable, objective methods to verify whether product packaging is truly eco-friendly.  
Manual audits are slow, inconsistent, and dependent on human perception.  

**EcoVision** solves this by using **computer vision and data science** to:
- Visually analyze packaging images.
- Classify them as **Eco-Friendly** or **Non-Sustainable**.
- Provide explainable AI visualizations (Grad-CAM) and sustainability insights.

---

##  Features

 **Computer Vision Classification** – Deep learning model detects eco vs. non-eco packaging.  
 **Transfer Learning** – Uses pre-trained models (MobileNetV2 / EfficientNetB0) for high accuracy.  
 **Explainable AI (Grad-CAM)** – Visualizes the regions influencing AI predictions.  
 **Sustainability Dashboard** – Shows metrics like Eco Compliance %, Confidence, and Trend Analysis.  
 **Streamlit App Interface** – Upload product image → get prediction → see visual explanation instantly.

---

##  Tech Stack

| Domain | Tools / Frameworks |
|---------|--------------------|
| **Deep Learning** | TensorFlow, Keras |
| **Computer Vision** | OpenCV, Grad-CAM |
| **Data Science & Visualization** | Pandas, Matplotlib, Seaborn, Plotly |
| **Web Deployment** | Streamlit  |
| **Dataset Source** | [Kaggle – Environmental Images for Sustainability](https://www.kaggle.com/datasets/hamzaboulahia/environmental-images-for-sustainability) |

---

## Project Workflow

1. **Data Collection & Preprocessing**
   - Load Kaggle dataset (eco-friendly vs. non-sustainable packaging).
   - Resize, normalize, and augment images.

2. **Model Development**
   - Build baseline CNN model.
   - Fine-tune Transfer Learning model (MobileNetV2).
   - Train using binary crossentropy and Adam optimizer.

3. **Evaluation & Explainability**
   - Evaluate with accuracy, F1-score, and confusion matrix.
   - Generate Grad-CAM visualizations for explainable AI.

4. **Sustainability Analytics**
   - Calculate Eco Compliance %, Average Confidence, and Material Insights.

5. **Deployment**
   - Build interactive Streamlit web app for live demo.

---

##  Impact

**EcoVision** supports the global shift towards **responsible retail and sustainable consumption** by:
- Reducing manual effort in sustainability audits.
- Empowering brands with AI-based ESG verification.
- Encouraging consumers to make greener purchasing decisions.

---

##  Future Scope

 Extend classification into multiple packaging categories (Paper, Plastic, Compostable).  
 Integrate OCR to detect eco-label text.  
 Add carbon footprint estimation based on predicted packaging material.  
 Deploy live dashboard for corporate ESG analytics.

---

##  Author

**Preethikgha M**  
 Linkedln - https://www.linkedin.com/in/preethikgha-m-7421b3313/

---

