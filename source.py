import pandas as pd
import numpy as np
import re
import os
import sys
import sqlite3
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from sklearn.model_selection import train_test_split
import logging
import multiprocessing
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
#from IPython.display import display
import pymorphy2
from sklearn import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score
morph = pymorphy2.MorphAnalyzer(lang = 'ru')
stop_words = stopwords.words('russian')
from sklearn.linear_model import SGDClassifier 
from scipy import sparse
from scipy.sparse import csr_matrix

# предобработка текста, удаление стоп-слов, токенизация, лемматизация
def preprocess_text(text, grams):
    
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
    text = re.sub('[^а-яА-Я()]+', ' ', text)
    
    text = re.sub('@[^\s]+', ' ', text)
    text = re.sub('#[^\s]+', ' ', text)

    # токенизация
    tokens = nltk.word_tokenize(text)
    #удаление стоп слов
    text = [word for word in tokens if word not in stop_words ]


    #лемматизация 
    text = [morph.parse(w)[0].normal_form for w in  text]

    if grams == 1 :
       text = ' '.join(text)
    else :
       text = list(ngrams(text, grams))
       text = [' '.join(w) for w in text]
       text = ' '.join(text) 
       

    return text





# загрузка обработанных данных в датасет

print('Грузим обработанный датасет)...')
try: 
    data_unigram = pd.read_csv("unigram_preprocessed.csv", encoding ='utf8')
    data_bigram = pd.read_csv("bigram_preprocessed.csv", encoding ='utf8')
    data_threegram = pd.read_csv("threegram_preprocessed.csv", encoding ='utf8')
except FileNotFoundError:

    n = ['1', '2', 'user', 'text', 'value', '6', '7', '8', '9', '10', '11', '12']
    data_positive = pd.read_csv('datasets/positive.csv', encoding ='utf8', sep=';', names =["1","2","3","text","value","6","7","8","9","10","11","12"], usecols=['text'] )
    data_negative = pd.read_csv('datasets/negative.csv', encoding ='utf8', sep=';', names =["1","2","3","text","value","6","7","8","9","10","11","12"], usecols=['text'])
    sample_size = min(data_positive.shape[0], data_negative.shape[0])
    raw_data = np.concatenate((data_positive['text'].values[:sample_size], data_negative['text'].values[:sample_size]), axis=0)
      
    labels = [1] * sample_size + [0] * sample_size   # список окрасов
    data_unigram = pd.DataFrame(data=None, index=None, columns={"review" ,"label", "preprocessed_review"}, dtype=None, copy=False)
    data_unigram['review'] = raw_data
    data_unigram['label'] = labels
    data_unigram.to_csv("prep1.csv", index = False)
    prep1 = pd.read_csv("prep1.csv", chunksize = 10000)

    data_bigram = pd.DataFrame(data=None, index=None, columns={"review" ,"label", "preprocessed_review"}, dtype=None, copy=False)
    data_bigram['review'] = raw_data
    data_bigram['label'] = labels
    data_bigram.to_csv("prep2.csv", index = False)
    prep2 = pd.read_csv("prep2.csv", chunksize = 10000)

    data_threegram = pd.DataFrame(data=None, index=None, columns={"review" ,"label", "preprocessed_review"}, dtype=None, copy=False)
    data_threegram['review'] = raw_data
    data_threegram['label'] = labels
    data_threegram.to_csv("prep3.csv", index = False)
    prep3 = pd.read_csv("prep3.csv", chunksize = 10000)

    chunk_list1 = []
    chunk_list2 = []
    chunk_list3 = []

    for data_chunk in prep1:
       filteredchunk1 = data_chunk['review'].apply(lambda review : preprocess_text(review ,1) )
       chunk_list1.append(filteredchunk1)

    d_u= pd.concat(chunk_list1)
    data_unigram['preprocessed_review']  = d_u.to_frame()
    data_unigram.to_csv("unigram_preprocessed.csv", index = False)


    for data_chunk in prep2:
       filteredchunk2 = data_chunk['review'].apply(lambda review : preprocess_text(review ,1) )
       chunk_list2.append(filteredchunk2)

    d_b= pd.concat(chunk_list2)
    data_bigram['preprocessed_review']  = d_b.to_frame()
    data_bigram.to_csv("bigram_preprocessed.csv", index = False)


    for data_chunk in prep3:
       filteredchunk3 = data_chunk['review'].apply(lambda review : preprocess_text(review ,1) )
       chunk_list3.append(filteredchunk3)

 
    d_t= pd.concat(chunk_list3)
    data_threegram['preprocessed_review']  = d_t.to_frame()
    data_threegram.to_csv("threegram_preprocessed.csv", index = False)

sample_size = len(data_unigram.index)




# загрузка модели

def load_model(filename):
    return joblib.load(filename)

# сохранение модели 
def save_model(filename, model):
    joblib.dump(model, filename)


    # векторизация методом мешка слов и построение модели
def model_bow(data, grams):
    d = data.copy()
    y = d['label'].values
    d.drop(['label'] , axis = 1, inplace = True)
    x_train, x_test, y_train, y_test = train_test_split(d, y, test_size=0.4, stratify = y, random_state = 0)
    vect = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'")
    X_train_review_bow = vect.fit_transform(x_train['preprocessed_review'].values.astype('U'))
    X_test_review_bow = vect.transform(x_test['preprocessed_review'].values.astype('U'))

    print(len(vect.get_feature_names()))

    clf = LogisticRegression(solver='sag', max_iter=100000)

    clf.fit(X_train_review_bow, y_train)

    y_pred = clf.predict(X_test_review_bow)

    print('f1_score : ', f1_score(y_test, y_pred, average='weighted'))

    # векторизация методом tf-idf и построение модели
def model_tfidf(data, grams):

    d = data.copy()
    y = d['label'].values
    d.drop(['label'] , axis = 1, inplace = True)
    x_train, x_test, y_train, y_test = train_test_split(d, y, test_size=0.4, stratify = y, random_state = 0)
    vectorizer = TfidfVectorizer()
    X_train_review_bow = vectorizer.fit_transform(x_train['preprocessed_review'].values.astype('U'))
    X_test_review_bow = vectorizer.transform(x_test['preprocessed_review'].values.astype('U'))
    clf = LogisticRegression(solver='sag', max_iter=10000)
    clf.fit(X_train_review_bow, y_train)
    y_pred = clf.predict(X_test_review_bow)
    print('Tf-idf', grams , '-grams :\n',  classification_report(y_test, y_pred))
    

#Тесты на датафреймах разной длины
def model_bow_T(data, grams, i = 0, j = 0):


    d = data.copy()[sample_size//2 - (i+1)*(sample_size//2//j)  : sample_size//2+ (i+1)*(sample_size//2//j)]

    y = d['label'].values

    d.drop(['label'] , axis = 1, inplace = True)

    x_train, x_test, y_train, y_test = train_test_split(d, y, test_size=0.4, stratify = y)
    vect = CountVectorizer(token_pattern=r"(?u)\b\w\w+\b|!|\?|\(|\)")


    X_train_review_bow = vect.fit_transform(x_train['preprocessed_review'].values.astype('U'))
    X_test_review_bow = vect.transform(x_test['preprocessed_review'].values.astype('U'))


    clf = LogisticRegression(solver='sag', max_iter=10000)

    clf.fit(X_train_review_bow, y_train)

    y_pred = clf.predict(X_test_review_bow)
    print('Test N ', i, ' Size = ', 2*((i+1)*(sample_size//2//j)))
    print('f1_score : ', f1_score(y_test, y_pred, average='macro'))
    
    print('Feature_size: ',len(vect.get_feature_names()))

def model_tfidf_T(data, grams, i = 0, j = 0):
 
    d = data.copy()[sample_size//2 - (i+1)*(sample_size//2//j)  : sample_size//2+ (i+1)*(sample_size//2//j)]

    y = d['label'].values

    d.drop(['label'] , axis = 1, inplace = True)

    x_train, x_test, y_train, y_test = train_test_split(d, y, test_size=0.4, stratify = y)

    vect = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b|!|\?|\(|\)")

    X_train_review_bow = vect.fit_transform(x_train['preprocessed_review'].values.astype('U'))
    X_test_review_bow = vect.transform(x_test['preprocessed_review'].values.astype('U'))



    clf = LogisticRegression(solver='sag', max_iter=10000)

    clf.fit(X_train_review_bow, y_train)

    y_pred = clf.predict(X_test_review_bow)
    print('Test N ', i, ' Size = ', 2*((i+1)*(sample_size//2//j)))
    print('f1_score : ', f1_score(y_test, y_pred, average='macro'))
    
    print('Feature_size: ',len(vect.get_feature_names()))


    
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#


# Сохранение векторов слов(разряженные матрицы)

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


# Создание и сохранение массивов тестовой и обучающей выборки
def Form_and_save_vectors(data, v_t=1, g_t=1):
    d = data.copy()

    y = d['label'].values

    d.drop(['label'] , axis = 1, inplace = True)

    x_train, x_test, y_train, y_test = train_test_split(d, y, test_size=0.2, stratify = y) #разбиение  0.8 : 0.4  => 4 : 1


    #Выбор способа векторизации
    if v_t == 1:
        vect = TfidfVectorizer()
    else:  
        vect = CountVectorizer()

    
    #Векторизация
    X_train_review_bow = vect.fit_transform(x_train['preprocessed_review'].values.astype('U'))
    X_test_review_bow = vect.transform(x_test['preprocessed_review'].values.astype('U'))
    

    
    #Формируем имена файлов
    name = 'trn_tst/' + str(v_t) + '_' + str(g_t) + '_'
    
    
    #сохраняем  X_train_review_bow, X_test_review_bow, y_train, y_test в соответствующие файлы
    
    # Первая цифра: способ векторизации 1-TfidfVectorizer, 2- CountVectorizer
    # Вторая цифра: способ разбиения на граммы 1-униграммы, 2- биграммы, 3 - триграммы

    name1 = name + 'X_train_review_bow'
    save_sparse_csr(name1, X_train_review_bow)

    name1 = name + 'X_test_review_bow'
    save_sparse_csr(name1, X_test_review_bow)

    name1 = name + 'y_train'
    np.save(name1, y_train)

    name1 = name + 'y_test'
    np.save(name1, y_test)



#Загружаем тестовые и обучающие выборки
def Load_test_and_train(v_t=1, g_t=1):

    name = 'trn_tst/' + str(v_t) + '_' + str(g_t) + '_'
    
    name1 = name + 'X_train_review_bow.npz'

    X_train_review_bow = load_sparse_csr(name1)

    name1 = name + 'X_test_review_bow.npz'
    X_test_review_bow = load_sparse_csr(name1)
    
    name1 = name + 'y_train.npy'
    y_train = np.load(name1)

    name1 = name + 'y_test.npy'
    y_test = np.load(name1)

    return X_train_review_bow, X_test_review_bow, y_train, y_test



def model_get(v_t, g_t, j=1):  # v_t - способ векторизации, g_t - способ разбиения на граммы, j - на сколько частей разбить выборку

    if v_t == 1:
        print('Vectorization TfidfVectorizer')
    else:
        print('Vectorization CountVectorizer')

    print('Gramms = ', g_t)
    print()
    x_train, x_test, y_train, y_test = Load_test_and_train(v_t, g_t) # загружаем выборки из файла
  

    #print('SGDClassifier         VS       LogisticRegression')

    length = len(y_train)//j # размер куска выборки

    #clf = SGDClassifier (alpha=.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=False, max_iter=100, verbose=0, tol=0.001)
    
    # Обучаем j моделей на выборках различной длины
    for s in range(0, j):     
        
        clf2 = LogisticRegression(solver='sag', max_iter=1000, tol = 0.1)

        #clf.partial_fit(X_train_review_bow[s*length  :  (s+1)*length], y_train[s*length  :  (s+1)*length], classes=np.unique( y_train))

        clf2.fit(x_train[0  :  (s+1)*length], y_train[0 :  (s+1)*length])

        #y_pred = clf.predict(X_test_review_bow)
        y_pred2 = clf2.predict(x_test)
        
        print('Test_size: ',  (s+1)*length)
        
        #print('f1_score : ', f1_score(y_test, y_pred, average='macro'), ' || ', f1_score(y_test, y_pred2, average='macro'))
        print('f1_score : ', f1_score(y_test, y_pred2, average='macro')) #выводим оценку
        print()




#Формируем выборки и сохраняем в файл (Прогнать 1 раз, потом закоментить можно)

#print('Создаем и сохраняем выборки')      
#Form_and_save_vectors(data_unigram, v_t = 1, g_t = 1)
#Form_and_save_vectors(data_bigram, v_t = 1, g_t = 2)
#Form_and_save_vectors(data_threegram, v_t = 1, g_t = 3)

#Form_and_save_vectors(data_unigram, v_t = 2, g_t = 1)
#Form_and_save_vectors(data_bigram, v_t = 2, g_t = 2)
#Form_and_save_vectors(data_threegram, v_t = 2, g_t = 3)


print('Приступаем к обучению')
#Обучаем, смотрим на результат
#С ростом размера обучающей выборки как правило растет точность модели

print()
print('----------------------------------------------------')
print('----------------------------------------------------')
print()
model_get(1 , 1,  5)  # TfidfVectorizer, униграммы, разбиваем выборки на 5 частей
print()
print('----------------------------------------------------')
print('----------------------------------------------------')
print()
model_get(1 , 2,  5)  # TfidfVectorizer, биграммы, разбиваем выборки на 5 частей
print()
print('----------------------------------------------------')
print('----------------------------------------------------')
print()
model_get(1 , 3,  5)  # TfidfVectorizer, триграммы, разбиваем выборки на 5 частей
print()
print('----------------------------------------------------')
print('----------------------------------------------------')
print()
model_get(2 , 1,  5)  # CountVectorizer, униграммы, разбиваем выборки на 5 частей
print()
print('----------------------------------------------------')
print('----------------------------------------------------')
print()
model_get(2 , 2,  5)  # CountVectorizer, биграммы, разбиваем выборки на 5 частей
print()
print('----------------------------------------------------')
print('----------------------------------------------------')
print()
model_get(2 , 3,  5)  # CountVectorizer, триграммы, разбиваем выборки на 5 частей

