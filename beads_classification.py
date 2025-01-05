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
data1 = pd.read_csv('./extracted_features_4um_03.csv')
data2 = pd.read_csv('./extracted_features_7um_03.csv')

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

# x_min, x_max = X_resampled[:, 0].min() - 0.1, X_resampled[:, 0].max() + 0.1
# y_min, y_max = X_resampled[:, 1].min() - 0.1, X_resampled[:, 1].max() + 0.1
# xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
#                      np.linspace(y_min, y_max, 200))
#
# Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.figure(figsize=(6, 5))
# plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
# plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled, cmap='coolwarm', edgecolor='k')
# plt.xlabel('Amplitude (V)')
# plt.ylabel('Transient Time (ms)')
# plt.title('Decision Boundary')
# plt.show()

plt.figure(figsize=(10, 5))

# Histogram for amplitude
plt.subplot(1, 2, 1)
plt.hist(data1_filtered['Amplitude'], bins=20, alpha=0.7, label='4µm', color='blue')
plt.hist(data2_filtered['Amplitude'], bins=20, alpha=0.7, label='7µm', color='red')
plt.xlabel('Amplitude (V)')
plt.ylabel('Frequency')
plt.title('Amplitude Distribution')
plt.legend()

# Histogram for transient time
plt.subplot(1, 2, 2)
plt.hist(data1_filtered['Width time (ms)'], bins=20, alpha=0.7, label='4µm', color='blue')
plt.hist(data2_filtered['Width time (ms)'], bins=20, alpha=0.7, label='7µm', color='red')
plt.xlabel('Transient Time (ms)')
plt.ylabel('Frequency')
plt.title('Transient Time Distribution')
plt.legend()

plt.tight_layout()
plt.show()
