# 📱 Used Smartphone Price Analytics  

## 📖 Overview  
This project explores **price prediction and classification in the used smartphone market**. With the rapid advancement of mobile technology, the resale market faces issues of **inconsistent pricing, buyer distrust, and lost profits**. Using **machine learning models**, this study provides data-driven insights for setting fair resale prices and classifying smartphones into **premium (High)** and **budget (Low)** categories.

The analysis uses **3,454 smartphone sales records** with 15 attributes, including **brand, operating system, screen size, memory, camera specifications, battery, days used, and prices (new and used)**.

---

## 🎯 Business Problem  
- Sellers face difficulty in setting fair resale prices, leading to **loss of profits**.  
- Buyers often encounter **overpriced or undervalued phones**, creating dissatisfaction and reduced trust.  
- Businesses lack clarity in **classifying devices** into premium vs. budget segments, resulting in missed revenue opportunities.  

---

## ✅ Business Goals  
1. Provide sellers with **consistent, competitive pricing** to increase profitability and attract more customers.  
2. Improve **market transparency** and build **customer trust**.  
3. Enable classification of phones into **High vs. Low price segments**, allowing differentiation of **premium phones** for higher margins.  

---

## 🔍 Analytical Goals & Approaches  
- **EDA (Exploratory Data Analysis):** Identify trends, patterns, and correlations between features.  
- **Handle Missing Data & Outliers:** Impute missing values with median-based grouping and filter unrealistic entries.  
- **Regression Models:** Estimate resale prices using **MLR, Decision Trees, Random Forest Regression**.  
- **Classification Models:** Categorize phones into **High vs. Low** using **Decision Tree (CART), Random Forest, Naïve Bayes**.  
- **Model Evaluation:** Compare models using RMSE, R² (for regression) and Accuracy, Precision, Recall, F1-score (for classification).  

---

## 📊 Dataset  
- **Size:** 3,454 smartphone records  
- **Features:** 15 attributes  
- **Target Variables:**  
  - `Normalized_Used_Price` (Regression)  
  - `Price_Class` (High / Low, Classification)  

**Attributes include:**  
`device_brand`, `os`, `screen_size`, `4g`, `5g`, `rear_camera_mp`, `front_camera_mp`, `internal_memory`, `ram`, `battery`, `weight`, `release_year`, `days_used`, `normalized_new_price`, `normalized_used_price`, `price_class`.

---

## ⚙️ Methodology  

### Data Preprocessing  
- **Missing Data:** Median imputation grouped by related attributes.  
- **Zero Values:** Checked and corrected unrealistic entries (e.g., zero cameras).  
- **Outliers:** Removed unrealistic values (e.g., >5500mAh battery, >26cm screen size, >253g weight).  

### Data Partitioning  
- **80% Training**, **20% Testing**  
- Stratified sampling ensures price distribution consistency.  

### Models Implemented  
1. **Regression**  
   - Multiple Linear Regression (MLR)  
   - Decision Tree Regression  
   - Random Forest Regression  
2. **Classification**  
   - Decision Tree Classification (CART)  
   - Random Forest Classification  
   - Naïve Bayes  

---

## 📈 Results  

### Regression Models  
| Model | RMSE | R² | Notes |  
|-------|------|----|-------|  
| MLR | 0.25 | 0.82 | Transparent & interpretable but misses complex interactions |  
| Decision Tree | 0.29 | 0.76 | Rule-based, interpretable, but less accurate |  
| Random Forest | **0.22** | **0.86** | Most accurate, robust, but less interpretable |  

### Classification Models  
| Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | Notes |  
|-------|----------|-----------|----------------------|-------------|-------|  
| Decision Tree | **90.2%** | 81.4% | 80.6% | 92.1% | Strong interpretability & balance |  
| Random Forest | 89.6% | 80.8% | 73.8% | 92.5% | Robust but misses some high-price phones |  
| Naïve Bayes | 85.2% | 75.7% | 74.3% | 89.8% | Transparent but weaker performance |  

---

## 💡 Key Insights  
- **New Phone Price** (`normalized_new_price`) is the strongest driver of resale value (~0.83 correlation).  
- **Hardware Specs** (battery, RAM, screen size, cameras) significantly affect resale pricing.  
- **Brand & OS:** Premium brands (Samsung, Apple) and newer OS (5G-enabled) retain higher resale value.  
- **Classification Models:** Decision Tree rules provide clear guidance for identifying **premium phones** (high storage, larger screens, higher weight, reputed brands).  

---

## 🚀 Business Impact  
- **For Sellers:** Competitive, consistent resale pricing; maximized profits from premium phones.  
- **For Buyers:** Transparency, avoiding overpaying for undervalued devices.  
- **For Businesses:** Better inventory segmentation, improved trade-in margins, and stronger customer trust.  

---


---

## 👨‍💻 Author  
**Reishekesh Reddy Inavola**  
Data Analytics, Webster University  

