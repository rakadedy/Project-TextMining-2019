#Import Libary yang dibutuhkan
from lxml import html
import requests
import re
import string
import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter


#Parsing Dokumen dari web Twitter
page = requests.get('https://twitter.com/zakwanzainal12')
tree = html.fromstring(page.content)

tweets = tree.xpath('//p[@class="TweetTextSize TweetTextSize--normal js-tweet-text tweet-text"]/text()')
Dokumen = "Elang dari danau toba sering terbang di sekitar sungai sumatra. Elang tersebut sering memangsa bersama-sama elang lainnya, dan mangsanya adalah ikan yang berada di sungai. Selain banyak elang, danau tersebut juga merupakan sumber air bersih sekaligus sebagai tempat wisata. Kemudian, diketahui pula populasi fauna yang terdapat pada sungai-sungai di sekitar danau juga sangat beragam, serta fauna-fauna tersebut memang dilestarikan. Oleh sebab itu, antara elang dan mangsa merupakan salah satu rantai makanan yang alami dan sudah mengakar dari cagar alam yang dilestarikan tersebut."


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
 
# Cleaning the text sentences so that punctuation marks, stop words &amp; digits are removed
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y

#Method untuk melakukan case folding (perubahan case)
def caseFolding(object):
    for var in range(len(object)):
        object[var] = object[var].lower()
    return object

#Method untuk menghilangkan Angka
def removeNumber(object):
    for var in range(len(object)):
        object[var] = re.sub(r'\d+','', object[var])
    return object

#Method untuk memfilter tanda baca
def removePunctuation(object):
    for var in range(len(object)):
        translation = object[var].maketrans("","",string.punctuation)
        object[var] = object[var].translate(translation)
    return object

#Method untuk melakukan Tokenisasi
def Lexing(object):
    for var in range(len(object)):
        object[var] = object[var].split()
    return object

#Method untuk melakukan StopWord Removal dengan Library NLTK Stopwords
def stopwordRemoval(object):
    listStopword = set(stopwords.words('indonesian'))
    for var in range(len(object)):
        max = len(object[var])
        for vars in range(max):
            if(max - 1 < vars):
                break
            if object[var][vars] in listStopword:
                del object[var][vars]
                max = max - 1
    return object

#Method untuk melakukan Stemming dengan Library Porter Stemmer
def stemming(object):
    st = PorterStemmer()
    stemmed=[]
    for var in range(len(object)):
        temp = st.stem(object[var])
        stemmed.append(temp)
    return stemmed

def preprocessing(object):
    return stopwordRemoval(Lexing(removeNumber(removePunctuation(caseFolding(object)))))

hasil = preprocessing(tweets)
print(hasil,"\n")

with open("Preprocessing.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(hasil)

print("Tweet yang diambil : ",len(tweets),"\n")

path = 'Preprocessing.csv'

train_clean_sentences = []
fp = open(path,'r')
for line in fp:
    line = line.strip()
    line = line.replace(","," ")
    train_clean_sentences.append(line)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_clean_sentences)
 
# Creating true labels for 26 training sentences
y_train = [0,1,1,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,0,1,1,0,1,0,1,0,0]

# Clustering the document with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=10)
modelknn.fit(X,y_train)
print (train_clean_sentences)


test_sentences = ["Tak perlu DSLR nak tangkap gambar secantik ini. Upah photographer RM200/300. Membazir. Snap gambar pakai phone je then edit cantik dh la dia. Cukup 20 retweet aku share cara dia. ",\
"Supremasi Kulit Putih lebih baik dari kulit hitam"]
 
test_clean_sentence = []
for test in test_sentences:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+","",cleaned)
    test_clean_sentence.append(cleaned)
 
Test = vectorizer.transform(test_clean_sentence)

print(test_clean_sentence)

true_test_labels = ['rasis','tidak']
predicted_labels_knn = modelknn.predict(Test)
 
print ("\nKalimat:\n1. ",\
test_sentences[0],"\n2. ",test_sentences[1])
print ("\n-------------------------------PREDICTIONS BY KNN------------------------------------------")
print ("\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_knn[0])],\
"\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_knn[1])])
 