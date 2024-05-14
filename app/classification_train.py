import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
import xgboost as xgb
from core.preprocessing import clear_text
import matplotlib.pyplot as plt

# print(os.getcwd())
# dataset_path = sys.argv[1]
# model_path = sys.argv[2]
static_assets_path = os.getcwd() + '/static'
datasets_path = static_assets_path + '/datasets'
models_path = static_assets_path + '/models'
misc_path = static_assets_path + '/misc'

dataset_file = datasets_path + '/imdb.csv'
vectorizer_path = misc_path + '/tfidf_vectorizer.pickle'
scaler_file = misc_path + '/min_max_scaler.pickle'
encoder_file = misc_path + '/label_encoder.pickle'
conf_matrix_img_file = '../conf_matrix.png'
metrics_file = misc_path + '/metrics.txt'
model_file = models_path + '/random_forest_model.pickle'

print('Reading dataset....')
df = pd.read_csv(dataset_file)
print('Cleanup corpus....')
df['review_cleaned'] = df['review'].tolist()

print('Vectorize corpus....')
tfidf = TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range=(1, 1),
                        analyzer='word',
                        tokenizer=None)

tfidf.fit(df['review_cleaned'])
X = tfidf.transform(df['review_cleaned'])

print('Scale features....')
scaler = MaxAbsScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

print('Encode labels....')
encoder = LabelEncoder()
encoder.fit(df['sentiment'])
Y = encoder.transform(df['sentiment'])

print('Split dataset....')
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=42)

print('Train model....')
param_grid = {
    'n_estimators': [100, 150, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [8],
    'random_state': [42]
}

forest_gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='roc_auc')
forest_gs.fit(X_train, Y_train)

model = forest_gs.best_estimator_

Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred, average="macro")
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

print(Y_test[0:10])
print(Y_pred[0:10])
print(encoder.classes_)

cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot()
plt.savefig(conf_matrix_img_file, dpi=120)


print('Storing vectorizer....')
pickle.dump(tfidf, open(vectorizer_path, 'wb'))
print('Storing scaler....')
pickle.dump(scaler, open(scaler_file, 'wb'))
print('Storing encoder....')
pickle.dump(encoder, open(encoder_file, 'wb'))
print('Storing model....')
pickle.dump(model, open(model_file, 'wb'))
print('All done...')
with open(metrics_file, "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}\n\n")

