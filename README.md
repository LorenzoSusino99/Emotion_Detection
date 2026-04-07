# Speech Emotion Recognition using Big Data Technologies

An end-to-end Big Data pipeline and machine learning project designed to classify human emotions from `.wav` audio files across three different languages: German, French, and English.

**Course**: Big Data  
**Professor**: Diego Angelo Gaetano Reforgiato Recupero  
**Authors**: Daniele Lurani, Lorenzo Susino

---

## About the Project
This project explores the intersection of audio signal processing and big data processing capabilities. By utilizing distributed computing frameworks, we parallelized the computationally expensive task of extracting audio features from thousands of `.wav` files. The extracted features were then used to train and evaluate supervised machine learning models to accurately classify the emotional state of the speaker.

## Tech Stack & Tools
* **Big Data Framework**: PySpark (for RDD parallelization and DataFrame management)
* **Audio Processing**: Librosa, Parselmouth
* **Machine Learning**: Scikit-Learn (Random Forest, Gradient Boosting, PCA, LDA)
* **Data Manipulation**: Pandas, NumPy
* **Environment**: Google Colab / Hadoop 3

---

## Methodology & Pipeline

### 1. Parallel Feature Extraction
Extracting data from audio signals sequentially is a major bottleneck. To resolve this, we utilized PySpark to load the dataset directories into Resilient Distributed Datasets (RDDs) and map partitions for parallel processing. 
For each audio file, we extracted a concatenated 32-feature vector consisting of:
* **MFCCs** (Mel-frequency cepstral coefficients) mean values
* **Chroma STFT** mean values
* **Spectral Contrast** mean values

### 2. Dimensionality Reduction Analysis
We experimented with dimensionality reduction techniques to optimize model performance:
* **PCA (Principal Component Analysis)**: Tested with 12 components, but ultimately discarded as empirical analysis showed it worsened overall model metrics.
* **LDA (Linear Discriminant Analysis)**: Tested with 2 components; while it sped up processing, it significantly lowered accuracy across all datasets compared to using the full feature set.

### 3. Machine Learning Models
We trained two primary ensemble models on a standard 80/20 train-test split:
* **Random Forest Classifier**
* **Gradient Boosting Classifier**

---

## Results and Performance
The models were evaluated independently on three separate linguistic datasets. The performance varied significantly depending on the language/dataset quality:

| Dataset | Model | Accuracy | Precision | F1-Score | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **German** | Random Forest | 78.02% | 78.53% | 77.01% | 0.9650 |
| **German** | Gradient Boosting | 78.02% | 78.82% | 77.73% | 0.9651 |
| **English** | Random Forest | 65.09% | 65.07% | 64.93% | 0.8855 |
| **English** | Gradient Boosting | 53.30% | 54.53% | 53.54% | 0.8712 |
| **French** | Random Forest | 35.38% | 37.23% | 34.27% | 0.7897 |
| **French** | Gradient Boosting | 33.85% | 35.14% | 33.23% | 0.7176 |

*(Note: The French dataset presented significant classification challenges, likely due to data imbalance, dataset size, or specific acoustic properties of the recording environment).*

---

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
