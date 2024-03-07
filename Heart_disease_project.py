
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


heart_data = pd.read_csv("C:/Users/Agniv/Desktop/Internships/devtern internship/machine learning project/Heart disease prediction project python/data set/archive/Heart_Disease_Prediction.csv")


heart_data.dropna(inplace=True)

null_columns = heart_data.columns[heart_data.isnull().any()]
null_counts = heart_data[null_columns].isnull().sum()
heart_data.drop(columns=null_columns, inplace=True)

numeric_data = heart_data.select_dtypes(include='number')


correlation_matrix = numeric_data.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Heart Disease Prediction Features')
plt.show()

X = heart_data.drop(columns=['Heart Disease'])
y = heart_data['Heart Disease']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, pos_label='Presence')
report = classification_report(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label='Presence')
f1 = f1_score(y_test, y_pred, pos_label='Presence')
matrix = confusion_matrix(y_test, y_pred)

print("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print("Predictions:")
print(predictions)
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(matrix)

coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coefficients.sort_values(by='Coefficient', ascending=False, inplace=True)
print("\nModel Coefficients:")
print(coefficients)



