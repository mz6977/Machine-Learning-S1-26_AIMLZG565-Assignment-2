# Machine Learning Assignment 2

## a. Problem statement

I trained five classification models and compared how well they predict diagnoses in the Wisconsin Diagnostic Breast Cancer dataset. I also made a Streamlit app where you can choose a model, view its results, or upload another labeled test CSV.

## b. Dataset description

The dataset has 569 records and 30 numeric features. The target has two classes: `0` means malignant and `1` means benign. The data comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) and is available through scikit-learn.

I used a stratified 80/20 split with random state 42. This gave 455 training records and 114 test records. The test records and their labels are in `test_data.csv`.

## c. GitHub Repository Link

[Machine-Learning-S1-26_AIMLZG565-Assignment-2](https://github.com/mz6977/Machine-Learning-S1-26_AIMLZG565-Assignment-2)

## Live app

[Open the Streamlit app](https://2026aj05852-machine-learning-assignment-2.streamlit.app/)

## d. Models used

The table shows each model's results on the same test data. For precision, recall, F1, and AUC, class `1` (benign) is treated as the positive class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9377 | 0.9692 | 0.8750 | 0.9197 | 0.8062 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | One of the strongest models: it tied for the best accuracy and had the highest AUC. It made one false positive and one false negative. |
| Decision Tree | This model had the lowest accuracy and missed more benign cases than the other models. |
| KNN | It found every benign case in the test set, but had a few more false positives than Logistic Regression and SVM. |
| Naive Bayes | It performed better than the Decision Tree, but its accuracy was lower than the other three models. |
| Support Vector Machine | Its results were close to Logistic Regression and tied it on accuracy, precision, recall, F1, and MCC. |
| **Overall winner** | **Logistic Regression**, based on its AUC and strong overall results on this test split. |

## Streamlit app

The live app compares all five saved models. Choose a model to view its metrics, confusion matrix, and classification report. You can also upload a CSV with the 30 feature columns and a `target` column. The app uses the saved models and does not train them again.

To run the project locally:

```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Files

- `app.py` — Streamlit app.
- `train_models.py` — script to split the data and train the models.
- `requirements.txt` — Python packages needed to run the project.
- `test_data.csv` — held-out test data.
- `model/` — saved models and dataset information.

## Deployment

The app is deployed on Streamlit Community Cloud. The GitHub repository and live app links are above. Add a screenshot of the assignment running in BITS Virtual Lab to the final submission PDF.
