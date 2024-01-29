import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Activity 1: Importing the libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
plt.style.use('fivethirtyeight')

# Activity 2: Read the Dataset
file_path = 'C:\\Users\\rajab\\Downloads\\dataset\\PS_20174392719_1491204439457_log.csv'
data = pd.read_csv(file_path)

# Dropping unnecessary columns
data.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)


# Displaying first five rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Displaying last five rows of the dataset
print("\nLast 5 rows of the dataset:")
print(data.tail())

# Correlation analysis using heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Activity 3: Object Data Label Encoding
label_encoder = LabelEncoder()
data['type_encoded'] = label_encoder.fit_transform(data['type'])
data.drop('type', axis=1, inplace=True)

# Activity 4: Data Preprocessing

# Activity 4.1: Checking for Null Values
print("Null values in the dataset:")
print(data.isnull().sum())

# Activity 4.2: Handling Outliers (Example using Boxplot for 'amount')
plt.figure(figsize=(8, 6))
sns.boxplot(x='amount', data=data)
plt.title('Boxplot for Amount Attribute')
plt.show()

# Activity 4.3: Handling Missing Values (if present)
# Example: Filling missing values with mean
data.fillna(data.mean(), inplace=True)


# Activity 4.4: Scaling or Normalization (if required)
# Example using Min-Max Scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop('isFraud', axis=1))  # Scale features except target

# Convert scaled data back to DataFrame (if necessary)
data_scaled_df = pd.DataFrame(data_scaled, columns=data.drop('isFraud', axis=1).columns)
# Example: Creating a new feature 'balance_difference'
data_scaled_df['balance_difference'] = data_scaled_df['newbalanceDest'] - data_scaled_df['oldbalanceOrg']

# Update features and target after preprocessing steps
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_encoded', 'balance_difference']
target = 'isFraud'


X = data_scaled_df[features]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['isFraud'])

# Splitting Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you have X_train, X_test, y_train, y_test ready for further model building steps
# Activity 1: Random Forest Classifier
def RandomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    
    # Evaluation
    print("Random Forest Classifier Evaluation:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    
    return rf

# Activity 2: Decision Tree Classifier
def DecisionTree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predictions = dt.predict(X_test)
    
    # Evaluation
    print("Decision Tree Classifier Evaluation:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    
    return dt

# Activity 3: Extra Trees Classifier
def ExtraTree(X_train, X_test, y_train, y_test):
    et = ExtraTreesClassifier()
    et.fit(X_train, y_train)
    predictions = et.predict(X_test)
    
    # Evaluation
    print("Extra Trees Classifier Evaluation:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    
    return et

# Activity 4: Support Vector Machine Classifier
def SupportVector(X_train, X_test, y_train, y_test):
    svc = SVC()
    svc.fit(X_train, y_train)
    predictions = svc.predict(X_test)
    
    # Evaluation
    print("Support Vector Machine Classifier Evaluation:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    
    return svc

# Activity 5: XGBoost Classifier
def xgboost(X_train, X_test, y_train, y_test):
    xg =XGBClassifier()
    xg.fit(X_train, y_train)
    predictions = xg.predict(X_test)
    
    # Evaluation
    print("XGBoost Classifier Evaluation:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    
    return xg
# Activity 6: Compare the Models
def compareModels(models, X_train, X_test, y_train, y_test):
    for name, model in models.items():
        print(f"Evaluating {name}:")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Evaluation
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

# Activity 7: Evaluating performance of the model and saving the model
def saveModel(model, X, y, filename):
    model.fit(X, y)
    pickle.dump(model, open(filename, 'wb'))
# Assuming X_train, X_test, y_train, y_test are available after preprocessing
rf_model = RandomForest(X_train, X_test, y_train, y_test)
dt_model = DecisionTree(X_train, X_test, y_train, y_test)
et_model = ExtraTree(X_train, X_test, y_train, y_test)
svc_model = SupportVector(X_train, X_test, y_train, y_test)
xg_model = xgboost(X_train, X_test, y_train, y_test)
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'Support Vector Machine': SVC(),
    'XGBoost': XGBClassifier()
}

compareModels(models, X_train, X_test, y_train, y_test)

# Save the best performing model
best_model = SVC()  # Assuming SVC performed the best
saveModel(best_model, X, y, "C:\\Users\\rajab\\Downloads\\dataset\\best_model2.pkl")