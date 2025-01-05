import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier


matplotlib.use("Qt5Agg")  # Use the Qt5Agg backend
# Load both datasets
data1 = pd.read_csv('./data/extracted_features_4um_03.csv')
data2 = pd.read_csv('./data/extracted_features_7um_03.csv')

# Filter for Channel 1 data only (as an example)
data1_filtered = data1[['Amplitude', 'Width time (ms)']].copy()
data2_filtered = data2[['Amplitude', 'Width time (ms)']].copy()
# data1_filtered['Amplitude'] = data1_filtered['Amplitude'].abs()  # Use absolute value
# data2_filtered['Amplitude'] = data2_filtered['Amplitude'].abs()  # Use absolute value

# Label each dataset
data1_filtered['Label'] = 0  # Label 0 for 7um dataset
data2_filtered['Label'] = 1  # Label 1 for 4um dataset

# Combine both datasets
combined_data = pd.concat([data1_filtered, data2_filtered])

# Separate features and labels
X = combined_data[['Amplitude', 'Width time (ms)']]
y = combined_data['Label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Then, split the resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize and train a Random Forest/SVM/kNN/or other types of classifier
# classifier = RandomForestClassifier(class_weight='balanced', random_state=42)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['4um', '7um']).plot()
# plt.title("Confusion Matrix for Classifying 4um and 7um Particles")
plt.show()

# ROC Curve
y_proba = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

