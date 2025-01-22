#------------------IMPORT DELLE LIBRERIE----------------
#ciaoflavioeheheheeheh
import pandas as pd # manipolazione e analisi dei dati tabulari
import numpy as np # calcolo numerico e la manipolazione di array multidimensionali
import matplotlib.pyplot as plt # per creare grafici personalizzati come istogrammi, scatter plot, e line plot
import seaborn as sns # grafici più complessi e accattivanti come heatmap, boxplot e pairplot
from sklearn.model_selection import train_test_split # Divide un dataset in dati di allenamento e di test in modo casuale
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # calcola il punteggio TF-IDF (Term Frequency-Inverse Document Frequency), che misura l'importanza relativa delle parole
from sklearn.naive_bayes import MultinomialNB # importa il modello MultinomialNB di Naive Bayes
from sklearn.metrics import classification_report, confusion_matrix
import os # Usata per interagire con il file system, ad esempio leggere directory o file
import re # Libreria standard di Python per manipolare stringhe e lavorare con espressioni regolari (regex)
import nltk #  l'elaborazione del linguaggio naturale (NLP)
from nltk.corpus import stopwords # Contiene elenchi di parole comuni in varie lingue (come "the", "and", "is"), spesso escluse durante l'analisi del testo
from nltk.stem import WordNetLemmatizer # Usato per ridurre le parole alle loro radici lessicali (es. "running" → "run")
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from xgboost import XGBClassifier # Classificatore basato sull'algoritmo di boosting, ottimizzato per prestazioni e velocità
from sklearn.svm import SVC # Support Vector Classifier: un modello di machine learning per classificazione che separa i dati con un margine massimo tra le classi
from bs4 import BeautifulSoup # Libreria per analizzare ed estrarre dati da file HTML e XML

#------------------IMPORT DEI DATASET------------------

easy_ham_path = 'db/easy_ham/easy_ham/' # Questa directory contiene e-mail "ham" (non spam) che sono relativamente facili da distinguere come non spam
hard_ham_path = 'db/hard_ham/hard_ham/' # Questa directory contiene e-mail "ham" che sono più difficili da distinguere
spam_path = 'db/spam_2/spam_2/' # Questa directory contiene e-mail di spam che il modello dovrà imparare a identificare

#-------------LETTURA E CREAZIONE DELLA LISTA-----------

def get_data_with_labels(path, label): # Questa funzione prende due parametri : path (percorso della directory che contiene i file delle email) e label (etichetta associata ai file nella directory)
    data = [] # Inizializza una lista vuota per memorizzare i dati delle e-mail
    files = os.listdir(path) # Elenca tutti i file nella directory specificata
    for file_name in files: # Per ogni file nella directory
        file_path = os.path.join(path, file_name) # Costruisce il percorso completo del file
        try:
            with open(file_path, encoding="ISO-8859-1") as f: # Prova ad aprire il file con il parametro di codifica ISO-8859-1, che è comune per file di testo non Unicode
                content = f.read() # Legge il contenuto del file
                data.append({'text': content, 'label': label}) # Salva il contenuto e l'etichetta in un dizionario e lo aggiunge alla lista data
        except Exception as e:
            print(f"Errore con il file {file_name}: {e}") # In caso di errore durante l'apertura o lettura del file, stampa un messaggio di errore
    return data

easy_ham = get_data_with_labels(easy_ham_path, 'ham') # contiene e-mail non spam dalla directory easy_ham_path con etichetta 'ham'
hard_ham = get_data_with_labels(hard_ham_path, 'ham') # contiene e-mail non spam più difficili dalla directory hard_ham_path con etichetta 'ham'
spam = get_data_with_labels(spam_path, 'spam') # contiene e-mail spam dalla directory spam_path con etichetta 'spam'

all_emails = pd.DataFrame(easy_ham + hard_ham + spam) # Combina le tre liste e pd.DataFrame converte la lista di dizionari in un DataFrame di Pandas per una gestione più comoda dei dati

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

def remove_email_headers(text): # Questa funzione rimuove gli header (intestazioni) delle e-mail
    lines = text.splitlines()
    filtered_lines = [line for line in lines if not re.match(r'^\s*(Received|Date|From|To|Subject|Message-ID):', line, re.IGNORECASE)]
    return '\n'.join(filtered_lines)

def extract_email_body(text): # Questa funzione estrae il corpo (contenuto principale) dell'e-mail
    # Usa un pattern per separare il corpo
    body_start = text.find("\n\n")  # Cerca la prima riga vuota che separa header e body
    if body_start != -1: # Se il separatore viene trovato
        return text[body_start:].strip()
    return text

# Funzione di preprocessing
def preprocess_text(text): # text è una stringa contenente il testo grezzo dell'e-mail
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

def plot_common_words(texts, title, n=20): # calcola le parole più comuni e i rispettivi conteggi
    words = ' '.join(texts).split() # Unisce tutti i testi e divide il testo in una lista di parole
    word_freq = Counter(words).most_common(n) # Counter per calcolare la frequenza di ciascuna parola
    words, counts = zip(*word_freq)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title(title)
    plt.xlabel('Frequenza')
    plt.ylabel('Parole')
    plt.show()

# Visualizza parole per email spam
spam_emails = all_emails[all_emails['label'] == 1]['cleaned_text'] # Seleziona le email con etichetta 1 (spam) dalla colonna cleaned_text, vengono filtrate e salvate in spam_emails
plot_common_words(spam_emails, "Parole più comuni nelle email spam")

# Visualizza parole per email non spam
non_spam_emails = all_emails[all_emails['label'] == 0]['cleaned_text'] # Seleziona le email con etichetta 0 (non spam) e salva il risultato in non_spam_emails
plot_common_words(non_spam_emails, "Parole più comuni nelle email non spam")

# Genera un WordCloud per spam
spam_words = ' '.join(spam_emails) # Combina tutte le email di spam in un'unica stringa
wordcloud_spam = WordCloud(width=800, height=400, background_color='black').generate(spam_words)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud per Email Spam")
plt.show()

# Genera un WordCloud per non spam
non_spam_words = ' '.join(non_spam_emails) # Combina tutte le email non spam in un'unica stringa
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
model_bag_naive = MultinomialNB() # Inizializza il classificatore Naive Bayes Multinomiale
model_bag_naive.fit(X_train_bag, y_train) # Addestra il modello usando il dataset di addestramento X_train_bag (matrice Bag of Words) e le etichette y_train
print("Accuratezza con Naive Bayes - Bag of Words:", model_bag_naive.score(X_test_bag, y_test))

y_pred_bag = model_bag_naive.predict(X_test_bag) # Effettua previsioni sull'insieme di test X_test_bag
print("Report di classificazione con Naive Bayes - Bag of Words:")
print(classification_report(y_test, y_pred_bag))

conf_matrix_bag = confusion_matrix(y_test, y_pred_bag) # Genera la matrice di confusione, che mostra:valori veri positivi (TP),Valori veri negativi (TN),falsi positivi (FP),falsi negativi (FN)
sns.heatmap(conf_matrix_bag, annot=True, fmt='d', cmap='Blues') # Visualizza la matrice di confusione in formato grafico, con annotazioni numeriche (annot=True)
plt.title('Matrice di Confusione con Naive Bayes - Bag of Words')
plt.show()

# Modello Naive Bayes con TF-IDF
print("Modello Naive Bayes con TF-IDF")
model_tfidf_naive = MultinomialNB() # Utilizza il classificatore Naive Bayes Multinomiale
model_tfidf_naive.fit(X_train_tfidf, y_train) # Addestra il modello sui dati di training trasformati con TF-ID
print("Accuratezza con Naive Bayes - TF-IDF:", model_tfidf_naive.score(X_test_tfidf, y_test))

y_pred_tfidf = model_tfidf_naive.predict(X_test_tfidf) # predict(X_test_tfidf) effettua previsioni sull'insieme di test basandosi sui dati trasformati in TF-IDF
print("Report di classificazione con Naive Bayes - TF-IDF:")
print(classification_report(y_test, y_pred_tfidf)) # Fornisce una valutazione dettagliata delle prestazioni per ogni classe

conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(conf_matrix_tfidf, annot=True, fmt='d', cmap='Oranges') # Utilizza sns.heatmap per rappresentare graficamente la matrice con annotazioni
plt.title('Matrice di Confusione con Naive Bayes - TF-IDF')
plt.show()


# Modello Random Forest con Bag of Words
print("Modello Random Forest con Bag of Words")
model_forest_bag = RandomForestClassifier() # Inizializza il modello Random Forest, un ensemble di alberi decisionali, molto potente per compiti di classificazione
model_forest_bag.fit(X_train_bag, y_train) # Allena il modello usando i dati di addestramento X_train_bag (matrice Bag of Words) e le etichette y_train
print("Accuratezza con Random Forest - Bag of Words:", model_forest_bag.score(X_test_bag, y_test)) # score(X_test_bag, y_test) calcola l'accuratezza del modello sui dati di test, confrontando le previsioni fatte dal modello con le etichette vere 

y_pred_bag = model_forest_bag.predict(X_test_bag) # predict(X_test_bag) esegue delle previsioni sui dati di test per determinare se ogni email è spam o non spam
print("Report di classificazione con Random Forest - Bag of Words:")
print(classification_report(y_test, y_pred_bag))

conf_matrix_bag = confusion_matrix(y_test, y_pred_bag) # Calcola la matrice di confusione che confronta le etichette reali con le previsioni fatte dal modello
sns.heatmap(conf_matrix_bag, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione con Random Forest - Bag of Words')
plt.show()

# Modello Random Forest con TF-IDF
print("Modello Random Forest con TF-IDF")
model_forest_tfidf = RandomForestClassifier() # Inizializza il classificatore Random Forest
model_forest_tfidf.fit(X_train_tfidf, y_train) # Addestra il modello sui dati trasformati in TF-IDF (X_train_tfidf) con le etichette corrispondenti y_train
print("Accuratezza con Random Forest - TF-IDF:", model_forest_tfidf.score(X_test_tfidf, y_test)) # Calcola la percentuale di etichette correttamente classificate nei dati di test trasformati in TF-IDF

y_pred_tfidf = model_forest_tfidf.predict(X_test_tfidf) #predict genera le previsioni sulle e-mail di test trasformate in TF-IDF assegnando a ciascuna e-mail un'etichetta predetta (spam o non spam)
print("Report di classificazione con Random Forest - TF-IDF:")
print(classification_report(y_test, y_pred_tfidf)) # Fornisce un report dettagliato sulle prestazioni del modello per ogni classe (spam e non spam)

conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf) # Genera la matrice di confusione
sns.heatmap(conf_matrix_tfidf, annot=True, fmt='d', cmap='Oranges') # Crea una rappresentazione visiva della matrice 
plt.title('Matrice di Confusione con Random Forest - TF-IDF')
plt.show()


# Modello XGBoost con Bag of Words
print("Modello XGBoost con Bag of Words")
model_xgb_bag = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') # XGBClassifier() è un modello di boosting che ottimizza le prestazioni combinando più alberi decisionali addrestati iterativamente, nelle parentesi i parametri utilizzati
model_xgb_bag.fit(X_train_bag, y_train) # Addestra il modello XGBoost utilizzando X_train_bag (caratteristiche rappresentate con Bag of Words) e y_train (etichette delle email spam e non spam)
print("Accuratezza con XGBoost - Bag of Words:", model_xgb_bag.score(X_test_bag, y_test)) # Calcola l'accuratezza (percentuale di previsioni corrette sul totale) del modello confrontando le previsioni con le etichette reali

y_pred_bag = model_xgb_bag.predict(X_test_bag) # Utilizza il modello addestrato per effettuare previsioni sulle email di test rappresentate con Bag of Words
print("Report di classificazione con XGBoost - Bag of Words:")
print(classification_report(y_test, y_pred_bag)) # Genera un riepilogo delle prestazioni del modello (Precisione,richiamo, F1-Score,supporto)

conf_matrix_bag = confusion_matrix(y_test, y_pred_bag)
sns.heatmap(conf_matrix_bag, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione con XGBoost - Bag of Words')
plt.show()

# Modello XGBoost con TF-IDF
print("Modello XGBoost con TF-IDF")
model_xgb_tfidf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss') # XGBClassifier() è un modello di boosting ottimizzato che costruisce alberi decisionali iterativamente per minimizzare la perdita
model_xgb_tfidf.fit(X_train_tfidf, y_train) # Addestra il modello XGBoost sui dati di training
print("Accuratezza con XGBoost - TF-IDF:", model_xgb_tfidf.score(X_test_tfidf, y_test)) #score calcola  l'accuratezza confrontando le previsioni con le etichette reali (mostra la percentuale di previsioni corrette sul totale)

y_pred_tfidf = model_xgb_tfidf.predict(X_test_tfidf) # Utilizza il modello addestrato per effettuare previsioni sui dati di test rappresentati con TF-IDF
print("Report di classificazione con XGBoost - TF-IDF:")
print(classification_report(y_test, y_pred_tfidf))

conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(conf_matrix_tfidf, annot=True, fmt='d', cmap='Oranges')
plt.title('Matrice di Confusione con XGBoost - TF-IDF')
plt.show()


# Modello SVM con Bag of Words
print("Modello SVM con Bag of Words")
model_svm_bag = SVC(kernel='linear', probability=True, random_state=42) # SVC() è una classe di Support Vector Classifier della libreria Scikit-learn
model_svm_bag.fit(X_train_bag, y_train) # Addestra il modello SVM
print("Accuratezza con SVM - Bag of Words:", model_svm_bag.score(X_test_bag, y_test)) # Calcola l'accuratezza confrontando le previsioni del modello con le etichette reali

y_pred_bag = model_svm_bag.predict(X_test_bag) # Utilizza il modello addestrato per fare previsioni sulle caratteristiche testate con Bag of Words
print("Report di classificazione con SVM - Bag of Words:")
print(classification_report(y_test, y_pred_bag)) 

conf_matrix_bag = confusion_matrix(y_test, y_pred_bag)
sns.heatmap(conf_matrix_bag, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice di Confusione con SVM - Bag of Words')
plt.show()

# Modello SVM con TF-IDF
print("Modello SVM con TF-IDF")
model_svm_tfidf = SVC(kernel='linear', probability=True, random_state=42)
model_svm_tfidf.fit(X_train_tfidf, y_train) # Addestra il modello utilizzando X_train_tfidf (caratteristiche rappresentate come TF-IDF) e y_train (etichette di classificazione spam o non spam)
print("Accuratezza con SVM - TF-IDF:", model_svm_tfidf.score(X_test_tfidf, y_test)) # Calcola l'accuratezza confrontando le previsioni con le etichette reali e restituisce la percentuale di previsioni corrette

y_pred_tfidf = model_svm_tfidf.predict(X_test_tfidf) # Effettua previsioni sui dati di test basati su TF-IDF
print("Report di classificazione con SVM - TF-IDF:")
print(classification_report(y_test, y_pred_tfidf))

conf_matrix_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(conf_matrix_tfidf, annot=True, fmt='d', cmap='Oranges')
plt.title('Matrice di Confusione con SVM - TF-IDF')
plt.show()


#------------------CONFRONTO DELLE ACCURATEZZE------------------

# Salviamo le accuratezze dei modelli
accuracies = { # Nel dizionario accuracies ogni chiave rappresenta un modello specifico con la tecnica di rappresentazione del testo (Bag of Words o TF-IDF) e i valori sono le accuratezze calcolate con score per ogni modello
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
