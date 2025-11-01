# 🏆 Amazon ML Challenge 2025 - Product Price Prediction
### A Student's Journey in Competitive ML

<div align="center">


[![Competition](https://img.shields.io/badge/Competition-Amazon%20ML%20Challenge-FF9900?style=for-the-badge&logo=amazon)](https://unstop.com)
[![Rank](https://img.shields.io/badge/Rank-183%2F10000%2B-blue?style=for-the-badge)](https://unstop.com)
[![Score](https://img.shields.io/badge/SMAPE-47.07-green?style=for-the-badge)](https://unstop.com)
[![Colab](https://img.shields.io/badge/Platform-Google%20Colab-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com)

**Building Multimodal AI on Free Resources**

[📊 View Results](#results) • [💡 What We Learned](#what-we-learned) • [🚧 Challenges Faced](#challenges-we-faced) • [👥 Team](#team)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Our Solution](#our-solution)
- [Results](#results)
- [What We Learned](#what-we-learned)
- [Challenges We Faced](#challenges-we-faced)
- [Why We Couldn't Reach Top 50](#why-we-couldnt-reach-top-50)
- [Installation & Usage](#installation--usage)
- [Team](#team)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This repository contains our **Rank 183** solution for the **Amazon ML Challenge 2025**. As students from Brainware University, we competed against 1000+ teams (including IITs and NITs) using only **free Google Colab resources**.

### Quick Stats

| Metric | Value |
|--------|-------|
| **Final Rank** | 183 / 10000+ teams |
| **Final Score** | 47.07 SMAPE |
| **Competition Duration** | 48 hours (Oct 11-13, 2025) |
| **Dataset Size** | 150K total samples |
| **Submission Limit** | 5 per day |
| **Compute Used** | Google Colab Free (T4 GPU) |
| **Total Cost** | ₹0 (completely free) |

---

## 🎯 Problem Statement

**Challenge:** Predict e-commerce product prices using multimodal data

**Input Data:**
- 📝 Product descriptions (catalog content)
- 🖼️ Product images (downloadable URLs)
- 📊 Item Pack Quantities (IPQ)

**Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)
- Lower is better (0-200% range)
- Top teams achieved ~41% SMAPE
- We achieved 47.07% SMAPE

---

## 💡 Our Solution

### 🧠 Triple Ensemble Architecture

We didn't rely on a single model. We built **three different AI models** and combined their predictions:

#### Model 1: Deep Learning (Multimodal Neural Network)
```
DistilBERT (Text) + EfficientNet-B2 (Images) + Feature Fusion
```
- **Text Understanding**: DistilBERT processes product descriptions
- **Image Analysis**: EfficientNet-B2 analyzes product photos
- **Feature Fusion**: Custom neural network combines everything

#### Model 2: XGBoost (Gradient Boosting)
```
30+ Engineered Features → Tree-based Learning → Price Prediction
```
- Excellent for structured/tabular data
- Fast training and inference
- Captures non-linear patterns

#### Model 3: LightGBM (Gradient Boosting)
```
Alternative perspective on patterns → Ensemble diversity
```
- Similar to XGBoost but different algorithm
- Provides "second opinion"
- Improves ensemble robustness

### 🎯 Final Prediction
```python
Final Price = (DL_prediction × weight1) + 
              (XGBoost_prediction × weight2) + 
              (LightGBM_prediction × weight3)

# Weights automatically calculated based on validation performance
```

---

## 🔧 Feature Engineering (30+ Features)

We didn't just feed raw data. We created smart features:

### Core Features
- **Product Measurements**: Value, unit, pack quantity
- **Total Volume**: Value × Pack Quantity (CRITICAL!)
- **Unit Types**: Fluid ounce, weight, count, etc.

### Brand & Category Intelligence
- **Brand Statistics**: Mean price, std, min, max for each brand
- **Category Patterns**: Average prices per category
- **Target Encoding**: Historical brand/category prices

### Quality Indicators
- **Premium Keywords**: Organic, luxury, gourmet, artisan
- **Certifications**: Gluten-free, vegan, kosher, non-GMO
- **Size Markers**: Bulk, mini, family pack

### Text Analytics
- **Statistics**: Length, word count, number count
- **Description Quality**: Has detailed description?
- **Numeric Extraction**: All numbers in product text

---

## 📊 Results

### 🏆 Final Standings

| Rank | Team | Institution | 
|------|------|-------------|
| 1 | MessI | Indian Institute of Technology (IIT), Madras |
| 2 | Zenith | Indian Institute of Engineering Science and Technology (IIEST), Shibpur | 
| 3 | 00 Team_Rocket | Indraprastha Institute of Information Technology (IIIT), Delhi |
| ... | ... | ... | ... | ... |
| **183** | **Code Crusher** | **Brainware University** |

### 📈 Our Performance Analysis

**What Our Score Means:**
- **47.07% SMAPE**: On average, our predictions were off by ~47%
- **Example**: If actual price = $100, we predicted $147 or $53
- **Gap from Top**: ~6 points (12% relative difference)

**Model-wise Performance:**
| Model | Validation SMAPE | Contribution |
|-------|-----------------|--------------|
| Deep Learning | ~15-18% | 35% |
| XGBoost | ~14-16% | 40% |
| LightGBM | ~15-17% | 25% |

**Reality Check:**
- ✅ Successfully implemented complex multimodal system
- ✅ Beat 800+ teams
- ❌ Significant gap between validation (15%) and test (47%)
- ❌ Clear overfitting issues

---

## 💡 What We Learned

### 🎓 Technical Skills Gained

#### 1. Multimodal AI
- ✅ Combining text, images, and structured data
- ✅ Using pre-trained models (BERT, EfficientNet)
- ✅ Building fusion architectures

#### 2. Ensemble Learning
- ✅ Weighted averaging strategies
- ✅ Model diversity importance
- ✅ Ensemble weight optimization

#### 3. Feature Engineering
- ✅ Domain-specific feature creation
- ✅ Target encoding techniques
- ✅ Text and numeric feature extraction

#### 4. Production ML
- ✅ End-to-end pipeline development
- ✅ Handling real-world messy data
- ✅ Working with limited resources

### 🧠 Soft Skills Developed

- **Time Management**: Delivering under 48-hour deadline
- **Teamwork**: Dividing work effectively among 4 members
- **Problem Solving**: Debugging at 2 AM with limited resources
- **Resource Optimization**: Making the most of free Colab
- **Resilience**: Dealing with failed submissions and errors

---

## 🚧 Challenges We Faced

### 1. 💰 Limited Computational Resources

**The Reality:**
- Used **Google Colab Free Tier** (no paid GPU access)
- **Session Limits**: 
  - 12-hour maximum runtime
  - Disconnects if idle for 90 minutes
  - Had to restart training multiple times
- **GPU Shortage**: Often no GPU available during peak hours
- **Memory Limits**: 15GB RAM (insufficient for full dataset)

**Impact:**
- Could only train for 8 epochs (top teams likely trained 15-20+)
- Had to reduce batch size (16 instead of 32-64)
- Couldn't experiment with larger models
- Limited hyperparameter tuning

### 2. ⏱️ Submission Limit (5 per day)

**The Constraint:**
- Only **5 submissions allowed per day**
- Each test run took 2-3 hours
- No room for trial-and-error

**Impact:**
- Couldn't test multiple strategies
- One bad submission = wasted opportunity
- Had to be extremely careful with each attempt
- Top teams likely had unlimited local GPU access

### 3. 🔄 Overfitting Problem

**What Happened:**
- Validation SMAPE: ~15%
- Test SMAPE: 47% ❌
- Clear sign of overfitting

**Why This Happened:**
- Single train-validation split (no cross-validation)
- Not enough training data diversity
- Limited time to implement proper regularization
- Couldn't afford to train 5-fold cross-validation (would take 5× longer)

### 4. 📥 Image Download Issues

**Problems:**
- 150,000 images to download
- Many URLs failed/throttled
- Took 6+ hours to download
- Some images corrupted

**Our Workaround:**
- Retry logic with exponential backoff
- Placeholder images for failed downloads
- But this affected model quality

### 5. ⚡ Training Time Constraints

**Timeline Breakdown:**
```
Hour 0-6:   Understanding problem + EDA
Hour 6-12:  Feature engineering
Hour 12-18: Downloading images (interrupted 3 times!)
Hour 18-30: Building models
Hour 30-42: Training (restarted 4 times due to disconnects)
Hour 42-48: Ensemble + Final submission
```

**What We Couldn't Do:**
- Extensive hyperparameter tuning
- Multiple model architectures
- Cross-validation
- Advanced data augmentation
- Large ensemble (would need more compute)

---

## 🎯 Why We Couldn't Reach Top 50

Let's be honest about the gap:

### 🏆 What Top 50 Teams Had

#### 1. **Better Compute Resources** 🖥️
```
Top Teams:              Us:
✅ Local GPU clusters   ❌ Free Colab (12hr limit)
✅ Multiple GPUs        ❌ Single T4 GPU
✅ 64GB+ RAM            ❌ 15GB RAM
✅ Unlimited training   ❌ Session disconnects
✅ 100+ experiments     ❌ 5-10 attempts
```

#### 2. **More Training Time** ⏰
```
Top Teams:              Us:
✅ 15-20 epochs         ❌ 8 epochs
✅ Batch size 64-128    ❌ Batch size 16
✅ Full dataset         ❌ Had to subsample
✅ 48+ hours training   ❌ 12 hours actual training
```

#### 3. **Advanced Techniques** 🧪
```
Top Teams:              Us:
✅ 10-fold CV           ❌ Single train-val split
✅ CLIP models          ❌ DistilBERT + EfficientNet
✅ Pseudo-labeling      ❌ Standard training
✅ Test-time augment    ❌ Basic augmentation
✅ Model stacking       ❌ Simple weighted avg
```

#### 4. **Experience & Resources** 🎓
```
Top Teams:              Us:
✅ IIT/NIT students     ❌ Tier-3 college
✅ Research lab access  ❌ Personal laptops
✅ Professor guidance   ❌ Self-learned
✅ Previous competitions ❌ First major ML hackathon
✅ Paid cloud credits   ❌ ₹0 budget
```

### 📊 The Performance Gap Analysis

| Factor | Top 50 | Us | Impact on Score |
|--------|--------|----|--------------------|
| Training Epochs | 15-20 | 8 | -2-3 points |
| Cross-Validation | 5-10 fold | None | -1-2 points |
| Model Size | Large | Medium | -1 point |
| Compute Time | Unlimited | Limited | -1-2 points |
| Hyperparameter Tuning | Extensive | Minimal | -1 point |
| **Total Estimated Gap** | | | **-6-9 points** |

### 🎯 Our Actual Gap: 5.78 points

**This analysis shows:**
- ✅ Our approach was fundamentally sound
- ✅ Implementation quality was good
- ❌ Resources were the main bottleneck
- ❌ Not enough time for optimization

---

## 🌟 What We're Proud Of

Despite the challenges:

### ✨ Achievements

1. ✅ **Ranked 183/10000+** using free resources
2. ✅ **Beat 90% of teams** including many from better colleges
3. ✅ **Built production-grade code** in 48 hours
4. ✅ **Implemented complex multimodal system** as undergrads
5. ✅ **Learned more in 2 days** than in 2 months of classes
6. ✅ **Complete documentation** (this README!)
7. ✅ **Zero cost** - proved skills matter more than resources

### 📈 Growth Metrics

**Before Hackathon:**
- Basic knowledge of ML algorithms
- Never trained a deep learning model
- No experience with ensembles

**After Hackathon:**
- ✅ Multimodal AI understanding
- ✅ Production deployment skills
- ✅ Competitive ML mindset
- ✅ Resource optimization mastery

---

## 🚀 Installation & Usage

### Prerequisites
```bash
# All you need
- Google Account (for Colab)
- Internet connection
- 2GB+ free space (for dataset)
```

### Quick Start (3 Steps)

#### Step 1: Open in Colab
```bash
1. Upload multimodal_price_colab.ipynb to Google Drive
2. Right-click → Open with Google Colaboratory
3. Click "Copy to Drive" to save your own version
```

#### Step 2: Upload Dataset
```bash
1. Upload train.csv and test.csv to Google Drive
2. Update paths in notebook Section 2:
   BASE_PATH = '/content/drive/MyDrive/YOUR_FOLDER'
```

#### Step 3: Run!
```bash
1. Runtime → Run all (or run cell by cell)
2. Wait ~6-8 hours for complete training
3. Download test_out.csv from results folder
```

### ⚠️ Important Notes

**If Colab Disconnects:**
```python
# Models auto-save after each epoch
# Just reload the last checkpoint and continue

# In Section 11, add:
if os.path.exists(f'{MODEL_SAVE_PATH}/checkpoint.pth'):
    model.load_state_dict(torch.load(f'{MODEL_SAVE_PATH}/checkpoint.pth'))
    print("Resumed from checkpoint!")
```

**To Reduce Training Time:**
```python
# Option 1: Train on subset
train_df = train_df.sample(n=30000, random_state=42)

# Option 2: Reduce epochs
NUM_EPOCHS = 5  # Instead of 8

# Option 3: Skip image training
# Comment out image download section
```

---

## 📦 Dependencies
```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
timm>=0.9.0
xgboost>=2.0.0
lightgbm>=4.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
Pillow>=10.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

**Total Installation Time:** ~5 minutes on Colab

---

## 🎓 For Students Learning From This

### 💡 Key Takeaways

#### 1. You Don't Need Expensive Hardware
- We used 100% free resources
- Focus on smart approaches, not brute force
- Cloud platforms democratize ML

#### 2. Feature Engineering > Model Complexity
- 30+ smart features beat fancy models
- Domain knowledge is crucial
- Simple features often work best

#### 3. Ensemble Everything
- Multiple weak models > One strong model
- Diversity is key
- Simple averaging works surprisingly well

#### 4. Validation Strategy Matters Most
- Our biggest mistake: single split
- Always use cross-validation
- Watch for overfitting

#### 5. Real ML ≠ Kaggle Tutorials
- Messy data, encoding issues, missing values
- Debugging takes 50% of time
- Production code requires different skills

### 🔧 How to Improve Our Solution

If you have more resources:
```python
# 1. Cross-Validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. More Training
NUM_EPOCHS = 15  # Instead of 8

# 3. Larger Models
# Try: CLIP, ViT, BERT-large

# 4. Advanced Augmentation
# Heavy augmentation for images and text

# 5. Pseudo-Labeling
# Use test predictions to augment training

# 6. Test-Time Augmentation (TTA)
# Average predictions across augmented versions
```

---

## 👥 Team Code Crusher

<table>
  <tr>
    <td align="center">
      <b>Subham Acherjee</b><br>
      Team Lead & ML Engineer<br>
      <a href="https://linkedin.com/in/yourprofile">LinkedIn</a>
    </td>
    <td align="center">
      <b>Sayandip Bar</b><br>
      Feature Engineering<br>
      <a href="https://linkedin.com/in/yourprofile">LinkedIn</a>
    </td>
    <td align="center">
      <b>Abhishek Roy</b><br>
      Deep Learning<br>
      <a href="https://linkedin.com/in/yourprofile">LinkedIn</a>
    </td>
    <td align="center">
      <b>Debabrata Dey</b><br>
      Ensemble & Optimization<br>
      <a href="https://linkedin.com/in/yourprofile">LinkedIn</a>
    </td>
  </tr>
</table>

**Institution:** Brainware University, Kolkata  
**Department:** Computer Science & Engineering  
**Year:** 2024-25

---

## 🙏 Acknowledgments

- **Unstop** for organizing this amazing challenge
- **Amazon** for the real-world problem statement
- **Google Colab** for free GPU access (the real MVP! 🏆)
- **Open Source Community** for incredible tools
- **Our College** for supporting our participation
- **Fellow Competitors** for inspiring us to push harder

---

## 📞 Contact & Connect

Want to discuss ML, collaborate, or just chat?
## 👥 Team Code Crusher

| Name | Role | GitHub |
|------|------|--------|
| **Subham Acherjee** | Team Lead & ML Engineer | [GitHub](https://github.com/subham2023) |
| **Sayandip Bar** | Feature Engineering | [GitHub](https://github.com/Sayandip05) |
| **Abhishek Roy** | Deep Learning | [GitHub](https://github.com/Abhishek-Royy) |
| **Debabrata Dey** | Ensemble & Optimization | [GitHub](https://github.com/Debabrata7719) |

**Institution:** Brainware University, Kolkata  
**Department:** Computer Science & Engineering  
**Year:** 2024–25


## 🎯 Future Work

We're not stopping here! Plans for improvement:

- [ ] Implement 5-fold cross-validation
- [ ] Try CLIP for better multimodal fusion
- [ ] Add pseudo-labeling technique
- [ ] Experiment with larger models (if we get access to better GPU)
- [ ] Build a web demo for price prediction
- [ ] Participate in more competitions!

---

## 📜 License

MIT License - Feel free to learn from and build upon our work!
```
Copyright (c) 2025 Team Code Crusher

Permission is hereby granted, free of charge, to use, modify, 
and distribute this software for educational purposes.
```

---

## ⭐ Final Thoughts

**To Future Participants:**

Don't let lack of resources stop you. We proved that:
- 🆓 Free tools can compete with paid infrastructure
- 🧠 Smart approaches beat brute force
- 📚 Learning matters more than winning
- 🤝 Teamwork multiplies capabilities
- 💪 Persistence trumps perfection

**Our Ranking (183/10000+) with ₹0 budget is proof that in ML, intelligence > investment.**

---

<div align="center">

### 🌟 If this inspired you, give us a star! 🌟

**Made with ❤️, ☕, and countless Colab disconnects**

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer)

</div>
```

---
