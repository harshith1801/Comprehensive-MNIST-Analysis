# **Comprehensive MNIST Digit Classification Analysis**

This repository contains a multi-stage analysis of the MNIST handwritten digit dataset, progressing from foundational classification techniques to advanced dimensionality reduction. The project compares multiple models (KNN, SVM) and data representation strategies (Raw, PCA, Kernel PCA, UMAP) to identify the most effective and efficient approach for digit recognition.

```
## **Project Structure**

Comprehensive-MNIST-Analysis/  
│  
├── main.ipynb                  \# Main Jupyter Notebook with the full analysis narrative.  
├── README.md                   \# This overview file.  
├── requirements.txt            \# Required Python libraries to run the project.  
├── .gitignore                  \# Specifies files for Git to ignore.  
│  
├── src/                        \# Folder for modular, reusable Python code.  
│   ├── \_\_init\_\_.py  
│   ├── data\_loader.py          \# Functions for loading and preprocessing MNIST data.  
│   ├── models.py               \# Functions for training and evaluating classifiers.  
│   └── reduction.py            \# Functions for applying dimensionality reduction.  
│  
└── images/                     \# Folder containing all saved plots and visualizations.  
    ├── accuracy\_comparison\_all\_methods.png  
    ├── confusion\_matrix\_pca.png  
    └── ... (and other key result images)

```

## **Technologies & Tools**

* **Primary Language:** Python 3  
* **Core Libraries:**  
  * **Scikit-learn:** For PCA, SVM, KNN, and performance metrics.  
  * **UMAP:** For UMAP dimensionality reduction.  
  * **Pandas:** For data manipulation and results tabulation.  
  * **NumPy:** For numerical operations.  
  * **Matplotlib / Seaborn:** For data visualization.  
* **Development Environment:** Jupyter Notebook

## **Key Features & Analysis Stages**

This project is broken down into a logical progression of analysis:

1. **Foundational KNN Analysis:**  
   * Applies the K-Nearest Neighbors (KNN) algorithm to a subset of 6 digits.  
   * Analyzes the impact of training set size and compares distance metrics (Euclidean vs. Manhattan).  
   * Evaluates a novel cluster-based sampling method against standard random sampling.  
2. **Performance Enhancement with PCA:**  
   * Introduces Principal Component Analysis (PCA) to reduce data dimensionality while retaining \~90% of the variance.  
   * Compares the performance of both KNN and Support Vector Machine (SVM) classifiers on the transformed data to measure accuracy and efficiency improvements.  
3. **Advanced Dimensionality Reduction Comparison:**  
   * Conducts a deep-dive comparison using the full 10-digit dataset.  
   * Benchmarks an SVM classifier's performance across four data representations: Raw Pixels, PCA, Kernel PCA, and UMAP.

## **How to Run**

1. **Clone the Repository:**  
   git clone \[https://github.com/harshith1801/Comprehensive-MNIST-Analysis.git\](https://github.com/harshith1801/Comprehensive-MNIST-Analysis.git)  
   cd Comprehensive-MNIST-Analysis

2. **Set up a Virtual Environment (Recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3. **Install Dependencies:**  
   pip install \-r requirements.txt

4. Download the MNIST Dataset:  
   The code requires the four raw MNIST data files. Download them from Yann LeCun's official website and place them in the root project directory (Comprehensive-MNIST-Analysis/).  
   **Required files:**  
   * train-images-idx3-ubyte.gz  
   * train-labels-idx1-ubyte.gz  
   * t10k-images-idx3-ubyte.gz  
   * t10k-labels-idx1-ubyte.gz

**Important:** After downloading, you must **unzip** the files. You should have the four files with the \-ubyte extension in your folder.

5. Run the Jupyter Notebook:  
   Launch Jupyter and open the main.ipynb file to view the complete analysis.  
   jupyter notebook main.ipynb

## **Citations & References**

* **MNIST Dataset:** LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). The MNIST database of handwritten digits. [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)  
* **Scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.  
* **UMAP:** McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv preprint arXiv:1802.03426.