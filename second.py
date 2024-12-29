#------------------IMPORT DELLE LIBRERIE----------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
from bs4 import BeautifulSoup

#------------------IMPORT DEI DATASET------------------

easy_ham_path = 'db/easy_ham/easy_ham/'
hard_ham_path = 'db/hard_ham/hard_ham/'
spam_path = 'db/spam_2/spam_2/'

#-------------LETTURA E CREAZIONE DELLA LISTA-----------

def get_data_with_labels(path, label): 
    data = [] 
    files = os.listdir(path) 
    for file_name in files:
        file_path = os.path.join(path, file_name)
        try:
            with open(file_path, encoding="ISO-8859-1") as f: 
                content = f.read()
                data.append({'text': content, 'label': label})
        except Exception as e:
            print(f"Errore con il file {file_name}: {e}")
    return data

easy_ham = get_data_with_labels(easy_ham_path, 'ham')  
hard_ham = get_data_with_labels(hard_ham_path, 'ham')  
spam = get_data_with_labels(spam_path, 'spam')  

all_emails = pd.DataFrame(easy_ham + hard_ham + spam)

#------------------ANALISI DEI DATI------------------

sns.countplot(data=all_emails, x='label')
plt.title('Distribuzione delle Email')
plt.xlabel('Etichetta (Spam/Non Spam)')
plt.ylabel('Conteggio')
plt.show()


#------------------PREPROCESSING---------------------

nltk.download('stopwords') # Lista di parole comuni da rimuovere
nltk.download('wordnet') # Dizionario inglese
all_emails['label'] = all_emails['label'].map({'ham': 0, 'spam': 1})

def remove_email_headers(text):
    lines = text.splitlines()
    filtered_lines = [line for line in lines if not re.match(r'^\s*(Received|Date|From|To|Subject|Message-ID):', line, re.IGNORECASE)]
    return '\n'.join(filtered_lines)

def extract_email_body(text):
    # Usa un pattern per separare il corpo
    body_start = text.find("\n\n")  # Cerca la prima riga vuota che separa header e body
    if body_start != -1:
        return text[body_start:].strip()
    return text

# Funzione di preprocessing
def preprocess_text(text):
    text = remove_email_headers(text)
    text = extract_email_body(text)
    soup = BeautifulSoup(text, "html.parser")  # Rimuovi tag HTML
    text = soup.get_text() # Estrai testo
    text = text.lower()  # Converti in minuscolo
    text = re.sub(r'http[s]?://\S+', '', text)  # Rimuovi URL
    text = re.sub(r'\d+', '', text)  # Rimuovi numeri
    text = re.sub(r'[^\w\s]', '', text)  # Rimuovi punteggiatura
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Rimuovi stopwords
    lemmatizer = WordNetLemmatizer() # Inizializza lemmatizer
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()]) # Trasforma le parole alla loro forma base (es. "running" → "run")
    return text

# Applica il preprocessing creando una nuova colonna con il testo preprocessato
all_emails['cleaned_text'] = all_emails['text'].apply(preprocess_text)


#------------------VISUALIZZAZIONE------------------

def plot_common_words(texts, title, n=20):
    words = ' '.join(texts).split()
    word_freq = Counter(words).most_common(n)
    words, counts = zip(*word_freq)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(title)
    plt.xlabel('Frequenza')
    plt.ylabel('Parole')
    plt.show()

# Visualizza parole per email spam
spam_emails = all_emails[all_emails['label'] == 1]['cleaned_text']
plot_common_words(spam_emails, "Parole più comuni nelle email spam")

# Visualizza parole per email non spam
non_spam_emails = all_emails[all_emails['label'] == 0]['cleaned_text']
plot_common_words(non_spam_emails, "Parole più comuni nelle email non spam")

# Genera un WordCloud per spam
spam_words = ' '.join(spam_emails)
wordcloud_spam = WordCloud(width=800, height=400, background_color='black').generate(spam_words)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud per Email Spam")
plt.show()

# Genera un WordCloud per non spam
non_spam_words = ' '.join(non_spam_emails)
wordcloud_non_spam = WordCloud(width=800, height=400, background_color='white').generate(non_spam_words)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_non_spam, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud per Email Non Spam")
plt.show()


#------------------MODELLO DI CLASSIFICAZIONE------------------

# Dividi i dati in train e test
X_train, X_test, y_train, y_test = train_test_split(all_emails['cleaned_text'], all_emails['label'], test_size=0.2, random_state=42) # 80% train, 20% test

# Utilizzo di Bag of Words (Trasforma il testo in una matrice che conta la frequenza di ciascuna parola)
bag_vectorizer = CountVectorizer()
X_train_bag = bag_vectorizer.fit_transform(X_train)
X_test_bag = bag_vectorizer.transform(X_test)

# Utiizzo di TF-IDF (Trasforma il testo in una matrice che conta la frequenza di ciascuna parola pesata per la sua rarità)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


#----------------ADDESTRAMENTO E VALUTAZIONE----------------

# Modello Naive Bayes con Bag of Words
print("Modello Naive Bayes con Bag of Words")
model_bag_naive = MultinomialNB()
model_bag_naive.fit(X_train_bag, y_train)
print("Accuratezza con Naive Bayes - Bag of Words:", model_bag_naive.score(X_test_bag, y_test))

y_pred_bag = model_bag_naive.predict(X_test_bag)
print("Report di classificazione con Naive Bayes - Bag of Words:")
print(classification_report(y_test, y_pred_bag))

conf_matrix_bag = confusion_matrix(y_test, y_pred_bag)
sns.heatmap(conf_matrix_bag, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione con Naive Bayes - Bag of Words')
plt.show()

# Modello Naive Bayes con TF-IDF
print("Modello Naive Bayes con TF-IDF")
model_tfidf_naive = MultinomialNB()
model_tfidf_naive.fit(X_train_tfidf, y_train)
print("Accuratezza con Naive Bayes - TF-IDF:", model_tfidf_naive.score(X_test_tfidf, y_test))

y_pred_tfidf = model_tfidf_naive.predict(X_test_tfidf)
print("Report di classificazione con Naive Bayes - TF-IDF:")
print(classification_report(y_test, y_pred_tfidf))

conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(conf_matrix_tfidf, annot=True, fmt='d', cmap='Oranges')
plt.title('Matrice di Confusione con Naive Bayes - TF-IDF')
plt.show()


# Modello Random Forest con Bag of Words
print("Modello Random Forest con Bag of Words")
model_forest_bag = RandomForestClassifier()
model_forest_bag.fit(X_train_bag, y_train)
print("Accuratezza con Random Forest - Bag of Words:", model_forest_bag.score(X_test_bag, y_test))

y_pred_bag = model_forest_bag.predict(X_test_bag)
print("Report di classificazione con Random Forest - Bag of Words:")
print(classification_report(y_test, y_pred_bag))

conf_matrix_bag = confusion_matrix(y_test, y_pred_bag)
sns.heatmap(conf_matrix_bag, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione con Random Forest - Bag of Words')
plt.show()

# Modello Random Forest con TF-IDF
print("Modello Random Forest con TF-IDF")
model_forest_tfidf = RandomForestClassifier()
model_forest_tfidf.fit(X_train_tfidf, y_train)
print("Accuratezza con Random Forest - TF-IDF:", model_forest_tfidf.score(X_test_tfidf, y_test))

y_pred_tfidf = model_forest_tfidf.predict(X_test_tfidf)
print("Report di classificazione con Random Forest - TF-IDF:")
print(classification_report(y_test, y_pred_tfidf))

conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(conf_matrix_tfidf, annot=True, fmt='d', cmap='Oranges')
plt.title('Matrice di Confusione con Random Forest - TF-IDF')
plt.show()


# Modello XGBoost con Bag of Words
print("Modello XGBoost con Bag of Words")
model_xgb_bag = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model_xgb_bag.fit(X_train_bag, y_train)
print("Accuratezza con XGBoost - Bag of Words:", model_xgb_bag.score(X_test_bag, y_test))

y_pred_bag = model_xgb_bag.predict(X_test_bag)
print("Report di classificazione con XGBoost - Bag of Words:")
print(classification_report(y_test, y_pred_bag))

conf_matrix_bag = confusion_matrix(y_test, y_pred_bag)
sns.heatmap(conf_matrix_bag, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione con XGBoost - Bag of Words')
plt.show()

# Modello XGBoost con TF-IDF
print("Modello XGBoost con TF-IDF")
model_xgb_tfidf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model_xgb_tfidf.fit(X_train_tfidf, y_train)
print("Accuratezza con XGBoost - TF-IDF:", model_xgb_tfidf.score(X_test_tfidf, y_test))

y_pred_tfidf = model_xgb_tfidf.predict(X_test_tfidf)
print("Report di classificazione con XGBoost - TF-IDF:")
print(classification_report(y_test, y_pred_tfidf))

conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(conf_matrix_tfidf, annot=True, fmt='d', cmap='Oranges')
plt.title('Matrice di Confusione con XGBoost - TF-IDF')
plt.show()


# Modello SVM con Bag of Words
print("Modello SVM con Bag of Words")
model_svm_bag = SVC(kernel='linear', probability=True, random_state=42)
model_svm_bag.fit(X_train_bag, y_train)
print("Accuratezza con SVM - Bag of Words:", model_svm_bag.score(X_test_bag, y_test))

y_pred_bag = model_svm_bag.predict(X_test_bag)
print("Report di classificazione con SVM - Bag of Words:")
print(classification_report(y_test, y_pred_bag))

conf_matrix_bag = confusion_matrix(y_test, y_pred_bag)
sns.heatmap(conf_matrix_bag, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione con SVM - Bag of Words')
plt.show()

# Modello SVM con TF-IDF
print("Modello SVM con TF-IDF")
model_svm_tfidf = SVC(kernel='linear', probability=True, random_state=42)
model_svm_tfidf.fit(X_train_tfidf, y_train)
print("Accuratezza con SVM - TF-IDF:", model_svm_tfidf.score(X_test_tfidf, y_test))

y_pred_tfidf = model_svm_tfidf.predict(X_test_tfidf)
print("Report di classificazione con SVM - TF-IDF:")
print(classification_report(y_test, y_pred_tfidf))

conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(conf_matrix_tfidf, annot=True, fmt='d', cmap='Oranges')
plt.title('Matrice di Confusione con SVM - TF-IDF')
plt.show()


#------------------CONFRONTO DELLE ACCURATEZZE------------------

# Salviamo le accuratezze dei modelli
accuracies = {
    "Naive Bayes (Bag of Words)": model_bag_naive.score(X_test_bag, y_test),
    "Naive Bayes (TF-IDF)": model_tfidf_naive.score(X_test_tfidf, y_test),
    "Random Forest (Bag of Words)": model_forest_bag.score(X_test_bag, y_test),
    "Random Forest (TF-IDF)": model_forest_tfidf.score(X_test_tfidf, y_test),
    "XGBoost (Bag of Words)": model_xgb_bag.score(X_test_bag, y_test),
    "XGBoost (TF-IDF)": model_xgb_tfidf.score(X_test_tfidf, y_test),
    "SVM (Bag of Words)": model_svm_bag.score(X_test_bag, y_test),
    "SVM (TF-IDF)": model_svm_tfidf.score(X_test_tfidf, y_test),
}

plt.figure(figsize=(12, 6))
plt.barh(list(accuracies.keys()), list(accuracies.values()), color='skyblue')
plt.xlabel('Accuratezza')
plt.title('Confronto delle Accuratezze dei Modelli')
plt.xlim(0, 1)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()