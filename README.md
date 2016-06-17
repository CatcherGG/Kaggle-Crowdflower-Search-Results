
# Crowdflower Search Results Relevance

## Data Exploration:

#### Imports. (Nothing to see here)


```python
import nltk
import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestClassifier
from sklearn.linear_model import LogisticRegression,SGDRegressor,PassiveAggressiveRegressor,LassoLars
from sklearn.metrics import mean_squared_error,classification_report,accuracy_score
from sklearn.svm import SVR,SVC
from sklearn.cross_validation import StratifiedKFold
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import *
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import re
import gensim
from gensim import corpora
import cPickle as pickle
import math
import gc
import warnings
from collections import Counter
warnings.filterwarnings('ignore')
```

### Read the data.


```python
train = pd.read_csv("train.csv").fillna("")
test  = pd.read_csv("test.csv").fillna("")
```

Let see what hides behind the csvs...


```python
list(train.columns)
```




    ['id',
     'query',
     'product_title',
     'product_description',
     'median_relevance',
     'relevance_variance']




```python
list(test.columns)
```




    ['id', 'query', 'product_title', 'product_description']



## Pre-process.
Our pre-process was including 3 main steps:
1. Cleaning HTML.
2. Removing stop words.
3. Stemming.

In any case, We saved 3 versions of the given columns - Original version, After cleaning [Stop words & HTML], After stemming.


```python
token_pattern = re.compile(r"(?u)\b\w\w+\b")
```


```python
def removeUnnecessaryText(df):
    regex = re.compile("(.style\d+\s+{[\w \-:;\"\'\\#\(\)\\n]*}\s*(\\n)*)*(?P<text>.*)",re.MULTILINE)
    return df.apply(lambda x: regex.match(BeautifulSoup(x).get_text('\n','\t').replace('\n',' ')).group("text"))
```


```python
def removeStopwords(df):
    cachedStopWords = stopwords.words("english")
    return df.apply(lambda x: ' '.join([word for word in token_pattern.findall(x) if word not in cachedStopWords]))
```


```python
stemmer = SnowballStemmer("english")
def preprocess(df):
    #remove html and styles
    df["product_description"] = removeUnnecessaryText(df["product_description"])
    df["query"] = removeUnnecessaryText(df["query"])
    df["product_title"] = removeUnnecessaryText(df["product_title"])
    print "Done cleaning HTML"
    # removing stopwords
    df["product_description_clean"] = removeStopwords(df["product_description"])
    df["product_title_clean"] = removeStopwords(df["product_title"])
    df["query_clean"] = removeStopwords(df["query"])
    print "Done removing stopwords"
    # steming the words
    df["product_description_stemed"] = df["product_description_clean"].apply(lambda x: ' '.join([stemmer.stem(word) for word in token_pattern.findall(x)]))
    df["product_title_stemed"] = df["product_title_clean"].apply(lambda x: ' '.join([stemmer.stem(word) for word in token_pattern.findall(x)]))
    df["query_stemed"] = df["query_clean"].apply(lambda x: ' '.join([stemmer.stem(word) for word in token_pattern.findall(x)]))
    print "Done stemming"
    return df
```


```python
train = preprocess(train)
```

    Done cleaning HTML
    Done removing stopwords
    Done stemming
    

## Feature Engineering.


```python
train['median_relevance'].describe()
```




    count    10158.000000
    mean         3.309805
    std          0.980666
    min          1.000000
    25%          3.000000
    50%          4.000000
    75%          4.000000
    max          4.000000
    Name: median_relevance, dtype: float64



What we can see here is that most of the 'median revelance' are 4. [We'll use this information later on]

### Main Features:
1. BM25.
2. TFIDF.
3. N-GRAM - Diferences, Counts, ... (Word-Grams)
4. Original, Cleaned and Stemmed statistics [Lengths, Counts and differeneces].

Features we tried and didn't work good:
5. Digits.
6. Distances from train Mean lengths and RMS [Root mean square].


```python
class BM25 :
    def __init__(self, fn_docs, delimiter=' ') :
        self.dictionary = corpora.Dictionary()
        self.DF = {}
        self.delimiter = delimiter
        self.DocTF = []
        self.DocIDF = {}
        self.N = 0
        self.DocAvgLen = 0
        self.fn_docs = fn_docs
        self.DocLen = []
        self.buildDictionary()
        self.TFIDF_Generator()

    def buildDictionary(self) :
        all_docs = []
        for line in self.fn_docs:
            all_docs.append(line.strip().split(self.delimiter))
        self.dictionary.add_documents(all_docs)

    def TFIDF_Generator(self, base=math.e) :
        docTotalLen = 0
        for line in self.fn_docs :
            doc = line.strip().split(self.delimiter)
            docTotalLen += len(doc)
            self.DocLen.append(len(doc))
            #print self.dictionary.doc2bow(doc)
            bow = dict([(term, freq*1.0/len(doc)) for term, freq in self.dictionary.doc2bow(doc)])
            for term, tf in bow.items() :
                if term not in self.DF :
                    self.DF[term] = 0
                self.DF[term] += 1
            self.DocTF.append(bow)
            self.N = self.N + 1
        for term in self.DF:
            self.DocIDF[term] = math.log((self.N - self.DF[term] +0.5) / (self.DF[term] + 0.5), base)
        self.DocAvgLen = docTotalLen / self.N

    def BM25Score(self, Query=[], k1=1.5, b=0.75) :
        query_bow = self.dictionary.doc2bow(Query)
        scores = []
        for idx, doc in enumerate(self.DocTF) :
            commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
            tmp_score = []
            doc_terms_len = self.DocLen[idx]
            for term in commonTerms :
                upper = (doc[term] * (k1+1))
                below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.DocAvgLen))
                tmp_score.append(self.DocIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        return scores

    def TFIDF(self) :
        tfidf = []
        for doc in self.DocTF :
            doc_tfidf  = [(term, tf*self.DocIDF[term]) for term, tf in doc.items()]
            doc_tfidf.sort()
            tfidf.append(doc_tfidf)
        return tfidf

    def Items(self) :
        # Return a list [(term_idx, term_desc),]
        items = self.dictionary.items()
        items.sort()
        return items
```


```python
class FeatureMapper:
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        for feature_name, column_name, extractor in self.features:
            extractor.fit(X[column_name], y)

    def transform(self, X):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            #print column_name,feature_name
            fea = extractor.transform(X[column_name])
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]

    def fit_transform(self, X, y=None):
        extracted = []
        for feature_name, column_name, extractor in self.features:
            fea = extractor.fit_transform(X[column_name], y)
            if hasattr(fea, "toarray"):
                extracted.append(fea.toarray())
            else:
                extracted.append(fea)
        if len(extracted) > 1:
            return np.concatenate(extracted, axis=1)
        else: 
            return extracted[0]
```


```python
def identity(x):
    return x
```


```python
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to sklearn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]
```


```python
VECTOR_SIZE = 400
class SimilarityTransform(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        new_arr = []
        for row in X.toarray():
            bow_query = row[:VECTOR_SIZE]
            bow_title = row[VECTOR_SIZE:2*VECTOR_SIZE]
            bow_desc = row[2*VECTOR_SIZE:3*VECTOR_SIZE]
            cosine_query_title = cosine_similarity(bow_query,bow_title)[0]
            cosine_query_desc = cosine_similarity(bow_query,bow_desc)[0]
            cosine_title_desc = cosine_similarity(bow_title,bow_desc)[0]
            new_arr.append(list(row)+list(cosine_query_title)+list(cosine_query_desc)+list(cosine_title_desc))
        return new_arr
```


```python
class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T
```


```python
# calc n_grams
def getNGramsUnordered(n,string):
    word_lst = token_pattern.findall(string)
    grams = []
    for i in xrange(len(word_lst)-n+1):
        words = word_lst[i:i+n]
        for gram in itertools.permutations(words):
            grams.append('_'.join(list(gram)))
    return grams

def rms(lst):
    return np.sqrt(np.mean(np.square(lst)))

def mean_delta(lst):
    return abs((lst - lst.shift(-1)).mean())
```


```python
def extract_features(df):
    # count init texts lenghts
    df["query_init_len"] = df["query"].apply(lambda x: float(len(token_pattern.findall(x))) if x!='' else 0.0)
    df["desc_init_len"] = df["product_description"].apply(lambda x: float(len(token_pattern.findall(x))) if x!='' else 0.0)
    df["title_init_len"] = df["product_title"].apply(lambda x: float(len(token_pattern.findall(x))) if x!='' else 0.0)
    print "Done counting lenghts"
    
    # count stemmed texts length
    df["query_stemed_len"] = df["query_stemed"].apply(lambda x: float(len(token_pattern.findall(x))) if x!='' else 0.0)
    df["product_description_stemed_len"] = df["product_description_stemed"].apply(lambda x: float(len(token_pattern.findall(x))) if x!='' else 0.0)
    df["product_title_stemed_len"] = df["product_title_stemed"].apply(lambda x: float(len(token_pattern.findall(x))) if x!='' else 0.0)
    print "Done counting stemed lenghts"
    
    # difference between init and stemed&cleaned texts
    df["query_diff_len"] = df["query_init_len"]-df["query_stemed_len"]
    df["desc_diff_len"] = df["desc_init_len"]-df["product_description_stemed_len"]
    df["title_diff_len"] = df["title_init_len"]-df["product_title_stemed_len"]
    print "Done calculate length differences"
    
    # calc change ratio
    df["query_change_ratio"] = df["query_diff_len"]/df["query_init_len"]
    df["query_change_ratio"] = df["query_change_ratio"].replace([np.inf, -np.inf, np.nan],0)
    df["desc_change_ratio"] = df["desc_diff_len"]/df["desc_init_len"]
    df["desc_change_ratio"] = df["desc_change_ratio"].replace([np.inf, -np.inf, np.nan],0)
    df["title_change_ratio"] = df["title_diff_len"]/df["title_init_len"]
    df["title_change_ratio"] = df["title_change_ratio"].replace([np.inf, -np.inf, np.nan],0)
    print "Done calc change ratio"
    
    # calc length ratio
    df["query_title_ratio"] = df["query_stemed_len"]/df["product_title_stemed_len"]
    df["query_title_ratio"] = df["query_title_ratio"].replace([np.inf, -np.inf, np.nan],0)
    df["query_desc_ratio"] = df["query_stemed_len"]/df["product_description_stemed_len"]
    df["query_desc_ratio"] = df["query_desc_ratio"].replace([np.inf, -np.inf, np.nan],0)
    df["title_desc_ratio"] = df["product_title_stemed_len"]/df["product_description_stemed_len"]
    df["title_desc_ratio"] = df["title_desc_ratio"].replace([np.inf, -np.inf, np.nan],0)
    print "Done calc length ratio"
    
    # calc length ratio
    df["query_title_ratio"] = df["query_stemed_len"]/df["product_title_stemed_len"]
    df["query_title_ratio"] = df["query_title_ratio"].replace([np.inf, -np.inf, np.nan],0)
    df["query_desc_ratio"] = df["query_stemed_len"]/df["product_description_stemed_len"]
    df["query_desc_ratio"] = df["query_desc_ratio"].replace([np.inf, -np.inf, np.nan],0)
    df["title_desc_ratio"] = df["product_title_stemed_len"]/df["product_description_stemed_len"]
    df["title_desc_ratio"] = df["title_desc_ratio"].replace([np.inf, -np.inf, np.nan],0)
    print "Done calc length ratio"
       
    # empty description flag
    df["no_desc"] = df["product_description_stemed_len"]==0
    print "Done flaging empty description"
    
    # calc BM25
    for i, row in df.iterrows():
        bm25 = BM25([row["product_title_stemed"],row["product_description_stemed"]], delimiter=' ')
        Query = row["query_stemed"]
        Query = Query.split(' ')
        scores = bm25.BM25Score(Query)
        df.set_value(i,"BM25Title",scores[0])
        df.set_value(i,"BM25Description",scores[1])
    print "Done calc BM25"
    
    # calc number of similar and unsimiliar grams
    GRAMS = 3
    for n in range(1,GRAMS+1):
        #df["query_stemed_%sgram"%n] = df["query_stemed"].apply(lambda x: set(getNGramsUnordered(n,x)))
        #df["desc_stemed_%sgram"%n] = df["product_description_stemed"].apply(lambda x: set(getNGramsUnordered(n,x)))
        #df["title_stemed_%sgram"%n] = df["product_title_stemed"].apply(lambda x: set(getNGramsUnordered(n,x)))
        for i, row in df.iterrows():
            query_stemed_gram = set(getNGramsUnordered(n,row["query_stemed"]))
            title_stemed_gram = set(getNGramsUnordered(n,row["product_title_stemed"]))
            desc_stemed_gram = set(getNGramsUnordered(n,row["product_description_stemed"]))
            intersect1 = query_stemed_gram.intersection(title_stemed_gram)
            intersect2 = query_stemed_gram.intersection(desc_stemed_gram)
            intersect3 = title_stemed_gram.intersection(desc_stemed_gram)
            df.set_value(i,"query_title_similar_%sgram_len"%n,float(len(intersect1)))
            df.set_value(i,"query_desc_similar_%sgram_len"%n,float(len(intersect2)))
            df.set_value(i,"title_desc_similar_%sgram_len"%n,float(len(intersect3)))
            df.set_value(i,"query_title_similar_%sgram_percent"%n,float(len(intersect1))/row["product_title_stemed_len"] if row["product_title_stemed_len"]>0 else 0.0)
            df.set_value(i,"query_desc_similar_%sgram_percent"%n,float(len(intersect2))/row["product_description_stemed_len"] if row["product_description_stemed_len"]>0 else 0.0)
            df.set_value(i,"title_desc_similar_%sgram_percent"%n,float(len(intersect3))/row["product_description_stemed_len"] if row["product_description_stemed_len"]>0 else 0.0)
            # Differences
            difference_num1 = float(len(query_stemed_gram.difference(title_stemed_gram)))
            difference_num2 = float(len(query_stemed_gram.difference(desc_stemed_gram)))
            difference_num3 = float(len(title_stemed_gram.difference(desc_stemed_gram)))
            difference_num4 = float(len(title_stemed_gram.difference(query_stemed_gram)))
            difference_num5 = float(len(desc_stemed_gram.difference(query_stemed_gram)))
            difference_num6 = float(len(desc_stemed_gram.difference(title_stemed_gram)))
            sym_difference_num1 = float(len(query_stemed_gram.symmetric_difference(title_stemed_gram)))
            sym_difference_num2 = float(len(query_stemed_gram.symmetric_difference(desc_stemed_gram)))
            sym_difference_num3 = float(len(title_stemed_gram.symmetric_difference(desc_stemed_gram)))
            
            df.set_value(i,"query_title_diff_%sgram"%n, difference_num1)
            df.set_value(i,"query_desc_diff_%sgram"%n, difference_num2)
            df.set_value(i,"title_desc_diff_%sgram"%n, difference_num3)
            df.set_value(i,"title_query_diff_%sgram"%n, difference_num4)
            df.set_value(i,"desc_query_diff_%sgram"%n, difference_num5)
            df.set_value(i,"desc_title_diff_%sgram"%n, difference_num6)
            df.set_value(i,"query_title_sym_diff_%sgram"%n, sym_difference_num1)
            df.set_value(i,"query_desc_sym_diff_%sgram"%n, sym_difference_num2)
            df.set_value(i,"title_desc_sym_diff_%sgram"%n, sym_difference_num3)
            
            #
            
    print "Done calc similar words"
    
    """
    # digits
    for i, row in df.iterrows():
        query, title, desc = row["query"], row["product_title"], row["product_description"]

        num_of_digits_query = len([float(w) for w in query if w.isdecimal()])
        sum_of_digits_query = sum([float(w) for w in query if w.isdecimal()])
        num_of_digits_title = len([float(w) for w in title if w.isdecimal()])
        sum_of_digits_title = sum([float(w) for w in title if w.isdecimal()])
        num_of_digits_desc = len([float(w) for w in desc if w.isdecimal()])
        sum_of_digits_desc = sum([float(w) for w in desc if w.isdecimal()])
        num_of_unique_digits_query = len(set([float(w) for w in query if w.isdecimal()]))
        num_of_unique_digits_title = len(set([float(w) for w in title if w.isdecimal()]))
        num_of_unique_digits_desc = len(set([float(w) for w in desc if w.isdecimal()]))
        
        df.set_value(i,"num_of_digits_query", num_of_digits_query)
        df.set_value(i,"num_of_digits_title", num_of_digits_title)
        df.set_value(i,"num_of_digits_desc", num_of_digits_desc)
        
        df.set_value(i,"sum_of_digits_query", sum_of_digits_query)
        df.set_value(i,"sum_of_digits_title", sum_of_digits_title)
        df.set_value(i,"sum_of_digits_desc", sum_of_digits_desc)
        df.set_value(i,"num_of_unique_digits_query", num_of_unique_digits_query)
        df.set_value(i,"num_of_unique_digits_title", num_of_unique_digits_title)
        df.set_value(i,"num_of_unique_digits_desc", num_of_unique_digits_desc)
    
    
    print 'Done calc digits...'
    
    
    # Calc rms, avg
    for i, row in df.iterrows():
        local_train = train[train['query'] == row[1]]
        desc_len, title_len = row['product_description_stemed_len'], row['product_title_stemed_len']

        # distance from RMS
        desc_local_rms, desc_global_rms = rms(train['product_description_stemed_len']), rms(local_train['product_description_stemed_len'])
        title_local_rms, title_global_rms = rms(train['product_title_stemed_len']), rms(local_train['product_title_stemed_len'])

        desc_dist_local_rms, desc_dist_rms = desc_len - desc_local_rms, desc_len - desc_global_rms
        title_dist_local_rms, title_dist_rms = title_len - title_local_rms, title_len - title_global_rms

        df.set_value(i,"desc_dist_local_rms", desc_dist_local_rms)
        df.set_value(i,"desc_dist_rms", desc_dist_rms)
        df.set_value(i,"title_dist_local_rms", title_dist_local_rms)
        df.set_value(i,"title_dist_rms", title_dist_rms)
        
        # distance from Average
        desc_global_len_avg, desc_local_len_avg = train['product_description_stemed_len'].mean(), local_train['product_description_stemed_len'].mean()
        title_global_len_avg, title_local_len_avg = train['product_description_stemed_len'].mean(), local_train['product_description_stemed_len'].mean()

        desc_dist_local_avg, desc_dist_avg = desc_len - desc_local_len_avg, desc_len - desc_global_len_avg
        title_dist_local_avg, title_dist_avg = title_len - title_local_len_avg, title_len - title_global_len_avg
        
        df.set_value(i,"desc_dist_local_avg", desc_dist_local_avg)
        df.set_value(i,"desc_dist_avg", desc_dist_avg)
        df.set_value(i,"title_dist_local_avg", title_dist_local_avg)
        df.set_value(i,"title_dist_avg", title_dist_avg)
    
    print "Done average, RMS"
    """
    
    # calc local tfidf
    for i, row in df.iterrows():
        res = TfidfVectorizer(max_features=VECTOR_SIZE).fit_transform([row["query_stemed"],row["product_title_stemed"],row["product_description_stemed"]])
        # print res.getnnz(),res
    return df
```


```python
train = extract_features(train)
```

    Done counting lenghts
    Done counting stemed lenghts
    Done calculate length differences
    Done calc change ratio
    Done calc length ratio
    Done calc length ratio
    Done flaging empty description
    Done calc BM25
    Done calc similar words
    


```python
#                          Feature Set Name            Data Frame Column              Transformer
features = FeatureUnion(transformer_list=[('QueryBagOfWords',Pipeline([('selector', ItemSelector(key='query_stemed')),
                                                                      ('vectorizer',CountVectorizer(max_features=VECTOR_SIZE))])),
                                          ('TitleBagOfWords',Pipeline([('selector', ItemSelector(key='product_title_stemed')),
                                                                      ('vectorizer',CountVectorizer(max_features=VECTOR_SIZE))])),
                                          ('DescriptionBagOfWords',Pipeline([('selector', ItemSelector(key='product_description_stemed')),
                                                                      ('vectorizer',CountVectorizer(max_features=VECTOR_SIZE))])),
                                          ('QueryTFIDF',Pipeline([('selector', ItemSelector(key='query_stemed')),
                                                                      ('vectorizer',TfidfVectorizer(max_features=VECTOR_SIZE))])),
                                          ('TitleTFIDF',Pipeline([('selector', ItemSelector(key='product_title_stemed')),
                                                                      ('vectorizer',TfidfVectorizer(max_features=VECTOR_SIZE))])),
                                          ('DescriptionTFIDF',Pipeline([('selector', ItemSelector(key='product_description_stemed')),
                                                                      ('vectorizer',TfidfVectorizer(max_features=VECTOR_SIZE))])),
                                          ('QueryTokensInTitle1',Pipeline([('selector', ItemSelector(key='query_title_similar_1gram_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInDescription1',Pipeline([('selector', ItemSelector(key='query_desc_similar_1gram_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('TitleTokensInDescription1',Pipeline([('selector', ItemSelector(key='title_desc_similar_1gram_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInTitle2',Pipeline([('selector', ItemSelector(key='query_title_similar_2gram_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInDescription2',Pipeline([('selector', ItemSelector(key='query_desc_similar_2gram_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('TitleTokensInDescription2',Pipeline([('selector', ItemSelector(key='title_desc_similar_2gram_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInTitle3',Pipeline([('selector', ItemSelector(key='query_title_similar_3gram_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInDescription3',Pipeline([('selector', ItemSelector(key='query_desc_similar_3gram_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('TitleTokensInDescription3',Pipeline([('selector', ItemSelector(key='title_desc_similar_3gram_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInTitle1Percent',Pipeline([('selector', ItemSelector(key='query_title_similar_1gram_percent')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInDescription1Percent',Pipeline([('selector', ItemSelector(key='query_desc_similar_1gram_percent')),
                                                                      ('simple',SimpleTransform())])),
                                          ('TitleTokensInDescription1Percent',Pipeline([('selector', ItemSelector(key='title_desc_similar_1gram_percent')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInTitle2Percent',Pipeline([('selector', ItemSelector(key='query_title_similar_2gram_percent')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInDescription2Percent',Pipeline([('selector', ItemSelector(key='query_desc_similar_2gram_percent')),
                                                                      ('simple',SimpleTransform())])),
                                          ('TitleTokensInDescription2Percent',Pipeline([('selector', ItemSelector(key='title_desc_similar_2gram_percent')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInTitle3Percent',Pipeline([('selector', ItemSelector(key='query_title_similar_3gram_percent')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTokensInDescription3Percent',Pipeline([('selector', ItemSelector(key='query_desc_similar_3gram_percent')),
                                                                      ('simple',SimpleTransform())])),
                                          ('TitleTokensInDescription3Percent',Pipeline([('selector', ItemSelector(key='title_desc_similar_3gram_percent')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryBM25WithTitle',Pipeline([('selector', ItemSelector(key='BM25Title')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryBM25WithDesc',Pipeline([('selector', ItemSelector(key='BM25Description')),
                                                                      ('simple',SimpleTransform())])),
                                          ('DescriptionLength',Pipeline([('selector', ItemSelector(key='product_description_stemed_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('TitleLength',Pipeline([('selector', ItemSelector(key='product_title_stemed_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryLength',Pipeline([('selector', ItemSelector(key='query_stemed_len')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryTitleRatio',Pipeline([('selector', ItemSelector(key='query_title_ratio')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryDescRatio',Pipeline([('selector', ItemSelector(key='query_desc_ratio')),
                                                                      ('simple',SimpleTransform())])),
                                          ('TitleDescRatio',Pipeline([('selector', ItemSelector(key='title_desc_ratio')),
                                                                      ('simple',SimpleTransform())])),
                                          ('DescriptionCleanRatio',Pipeline([('selector', ItemSelector(key='desc_change_ratio')),
                                                                      ('simple',SimpleTransform())])),
                                          ('TitleCleanRatio',Pipeline([('selector', ItemSelector(key='title_change_ratio')),
                                                                      ('simple',SimpleTransform())])),
                                          ('QueryCleanRatio',Pipeline([('selector', ItemSelector(key='query_change_ratio')),
                                                                      ('simple',SimpleTransform())])),
                                          ('NoDescFlag',Pipeline([('selector', ItemSelector(key='no_desc')),
                                                                      ('simple',SimpleTransform())])),

                                          
                                          ####### 1 grram
                                          ('diff_1',Pipeline([('selector', ItemSelector(key='query_title_diff_1gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_2',Pipeline([('selector', ItemSelector(key='query_desc_diff_1gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_3',Pipeline([('selector', ItemSelector(key='title_desc_diff_1gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_4',Pipeline([('selector', ItemSelector(key='title_query_diff_1gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_5',Pipeline([('selector', ItemSelector(key='desc_query_diff_1gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_6',Pipeline([('selector', ItemSelector(key='desc_title_diff_1gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('sym_diff_1',Pipeline([('selector', ItemSelector(key='query_title_sym_diff_1gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('sym_diff_2',Pipeline([('selector', ItemSelector(key='query_desc_sym_diff_1gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('sym_diff_3',Pipeline([('selector', ItemSelector(key='title_desc_sym_diff_1gram')),
                                                                      ('simple',SimpleTransform())])),
                                          # 2 gram
                                          ('diff_1_2',Pipeline([('selector', ItemSelector(key='query_title_diff_2gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_2_2',Pipeline([('selector', ItemSelector(key='query_desc_diff_2gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_3_2',Pipeline([('selector', ItemSelector(key='title_desc_diff_2gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_4_2',Pipeline([('selector', ItemSelector(key='title_query_diff_2gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_5_2',Pipeline([('selector', ItemSelector(key='desc_query_diff_2gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_6_2',Pipeline([('selector', ItemSelector(key='desc_title_diff_2gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('sym_diff_1_2',Pipeline([('selector', ItemSelector(key='query_title_sym_diff_2gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('sym_diff_2_2',Pipeline([('selector', ItemSelector(key='query_desc_sym_diff_2gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('sym_diff_3_2',Pipeline([('selector', ItemSelector(key='title_desc_sym_diff_2gram')),
                                                                      ('simple',SimpleTransform())])),
                                          # 3_gram
                                          ('diff_1_3',Pipeline([('selector', ItemSelector(key='query_title_diff_3gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_2_3',Pipeline([('selector', ItemSelector(key='query_desc_diff_3gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_3_3',Pipeline([('selector', ItemSelector(key='title_desc_diff_3gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_4_3',Pipeline([('selector', ItemSelector(key='title_query_diff_3gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_5_3',Pipeline([('selector', ItemSelector(key='desc_query_diff_3gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('diff_6_3',Pipeline([('selector', ItemSelector(key='desc_title_diff_3gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('sym_diff_1_3',Pipeline([('selector', ItemSelector(key='query_title_sym_diff_3gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('sym_diff_2_3',Pipeline([('selector', ItemSelector(key='query_desc_sym_diff_3gram')),
                                                                      ('simple',SimpleTransform())])),
                                          ('sym_diff_3_3',Pipeline([('selector', ItemSelector(key='title_desc_sym_diff_3gram')),
                                                                      ('simple',SimpleTransform())]))
                                          
                                          

                                         ])
```


```python
# RMS, Average...
#('desc_dist_local_rms',Pipeline([('selector', ItemSelector(key='desc_dist_local_rms')),
#                          ('simple',SimpleTransform())])),
#('desc_dist_rms',Pipeline([('selector', ItemSelector(key='desc_dist_rms')),
#                          ('simple',SimpleTransform())])),
#('title_dist_local_rms',Pipeline([('selector', ItemSelector(key='title_dist_local_rms')),
#                          ('simple',SimpleTransform())])),
#('title_dist_rms',Pipeline([('selector', ItemSelector(key='title_dist_rms')),
#                          ('simple',SimpleTransform())])),
#('desc_dist_local_avg',Pipeline([('selector', ItemSelector(key='desc_dist_local_avg')),
#                          ('simple',SimpleTransform())])),
#('desc_dist_avg',Pipeline([('selector', ItemSelector(key='desc_dist_avg')),
#                          ('simple',SimpleTransform())])),
#('title_dist_local_avg',Pipeline([('selector', ItemSelector(key='title_dist_local_avg')),
#                          ('simple',SimpleTransform())])),
#('title_dist_avg',Pipeline([('selector', ItemSelector(key='title_dist_avg')),
#                         ('simple',SimpleTransform())])),

 # Digits
#('num_of_digits_query',Pipeline([('selector', ItemSelector(key='num_of_digits_query')),
#                          ('simple',SimpleTransform())])),
#('num_of_digits_title',Pipeline([('selector', ItemSelector(key='num_of_digits_title')),
#                          ('simple',SimpleTransform())])),
#('num_of_digits_desc',Pipeline([('selector', ItemSelector(key='num_of_digits_desc')),
#                          ('simple',SimpleTransform())])),
#('sum_of_digits_query',Pipeline([('selector', ItemSelector(key='sum_of_digits_query')),
#                          ('simple',SimpleTransform())])),
#('sum_of_digits_title',Pipeline([('selector', ItemSelector(key='sum_of_digits_title')),
#                          ('simple',SimpleTransform())])),
#('sum_of_digits_desc',Pipeline([('selector', ItemSelector(key='sum_of_digits_desc')),
#                          ('simple',SimpleTransform())])),
#('num_of_unique_digits_query',Pipeline([('selector', ItemSelector(key='num_of_unique_digits_query')),
#                          ('simple',SimpleTransform())])),
#('num_of_unique_digits_title',Pipeline([('selector', ItemSelector(key='num_of_unique_digits_title')),
#                          ('simple',SimpleTransform())])),
#('num_of_unique_digits_desc',Pipeline([('selector', ItemSelector(key='num_of_unique_digits_desc')),
#                          ('simple',SimpleTransform())]))
```

## Evaluation

At start We've tested all the classifiers below. [And it took about 8 hours to evaluate them all]
In ALL of our evaluations 'ExtraTreesRegressor' gave the best results, So we commented out all the other classifiers [To save time]


```python
clfs = {
    #"RandomForestRegressor":Pipeline([("extract_features", features),
    #                                  ('createMore',SimilarityTransform()),
    #                 ("regress", RandomForestRegressor(n_estimators=100,n_jobs=-1))]),
    #"RandomForestClassifier":Pipeline([("extract_features", features),
    #                                  ('createMore',SimilarityTransform()),
    #                 ("classify", RandomForestClassifier(n_estimators=100,n_jobs=-1))]),
    #"LogisticRegression":Pipeline([("extract_features", features),
    #                ('createMore',SimilarityTransform()),
    #                 ("regress", LogisticRegression())]),
    #"SGDRegressor":Pipeline([("extract_features", features),
    #                ('createMore',SimilarityTransform()),
    #                 ("regress", SGDRegressor())]),
    #"PassiveAggressiveRegressor":Pipeline([("extract_features", features),
    #                ('createMore',SimilarityTransform()),
    #                 ("regress", PassiveAggressiveRegressor())]),
    #"LassoLars":Pipeline([("extract_features", features),
    #                ('createMore',SimilarityTransform()),
    #                 ("regress", LassoLars())]),
    #'SVR':Pipeline([("extract_features", features),
    #                ('createMore',SimilarityTransform()),
    #                 ("regress", SVR())]),
    #'SVC':Pipeline([("extract_features", features),
    #                ('createMore',SimilarityTransform()),
    #                 ("classify", SVC())]),
    #'BaggingRegressor':Pipeline([("extract_features", features),
    #                            ('createMore',SimilarityTransform()),
    #                 ("regress", BaggingRegressor(n_jobs=-1))]),
    'ExtraTreesRegressor':Pipeline([("extract_features", features),
                                    ('createMore',SimilarityTransform()),
                     ("regress", ExtraTreesRegressor(n_estimators=100,n_jobs=-1))]),
    #'GradientBoostingRegressor':Pipeline([("extract_features", features),
    #                                      ('createMore',SimilarityTransform()),
    #                                     ("regress", GradientBoostingRegressor())])
}
```


```python
def predTrans(pred):
    return int(round(pred))
```


```python
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)
```

### Post-Process:
It is worth to mention that the classifier gave non-integer results and we had to find an optimal way to decide where to map it. So we tried 4 different approachs:
1. Floor.
2. Ceil.
3. Round.
4. CFD - Enforcing the train data distribution of median revelance on the predicted results.
The rank from worst to best in each of the times was: Floor (As worst), Ceil, Round, CFD (Best).


```python
def probabilities_median_table():
    coun = Counter(train['median_relevance'])
    max_feat = len(train['median_relevance'])
    return [float(median)/max_feat for median in coun.values()]

def prediction_distributions(pred):
    df = pd.DataFrame(data=pred, columns=['Prediction'])
    df.sort(columns=['Prediction'], ascending=True, inplace=True)
    probs = probabilities_median_table()
    
    num_of_one_median, num_of_two_median = int(math.floor(probs[0]*len(df))), int(math.floor(probs[1]*len(df)))
    num_of_three_median = int(math.floor(probs[2]*len(df)))
    
    for i, row in df.iterrows():
        df.set_value(i,"floor_prediction", math.floor(row['Prediction']))
        df.set_value(i,"ceil_prediction", math.ceil(row['Prediction']))
        df.set_value(i,"round_prediction", round(row['Prediction']))

        if num_of_one_median > 0:
            num_of_one_median -= 1
            df.set_value(i,"cfd_prediction", 1)
            continue
        if num_of_two_median > 0:
            num_of_two_median -= 1
            df.set_value(i,"cfd_prediction", 2)
            continue
        if num_of_three_median > 0:
            num_of_three_median -= 1
            df.set_value(i,"cfd_prediction", 3)
            continue

        df.set_value(i,"cfd_prediction", 4)
        
    df.sort_index(ascending=True, inplace=True)
    return df['cfd_prediction'].tolist(), df['round_prediction'].tolist(), df['ceil_prediction'].tolist(), df['floor_prediction'].tolist()
```


```python
NUMBER_OF_FOLDS = 3
kfold = StratifiedKFold(list(train["median_relevance"]),n_folds=NUMBER_OF_FOLDS)
def evaluateClf(clf):
    acc_score = []
    for train_index, test_index in kfold:
        X_train, X_test = train.iloc[train_index], train.iloc[test_index]
        y_train, y_test = train["median_relevance"].iloc[train_index], train["median_relevance"].iloc[test_index]
        clf.fit(X_train,y_train)
        preds = [pred for pred in clf.predict(X_test)]
        cfd_dest, round_dest, ceil_dest, floor_dest = prediction_distributions(preds)
        score_cfd = quadratic_weighted_kappa(y_test, cfd_dest)
        print "Score is: %s."%(score_cfd)
        acc_score.append(score_cfd)
    return clf,np.mean(acc_score),np.std(acc_score)
```


```python
#fitted_clfs = {}
def chooseBest():
    best_clf = None,0.0
    for name,clf in clfs.items():
        print "Evaluating %s"%name
        fitted_clf,avg_score,std = evaluateClf(clf)
        #fitted_clfs[name] = fitted_clf
        print "%s scored: %s with std: %s"%(name,avg_score,std)
        if best_clf[1]<avg_score:
            print "%s is currently the best"%(name)
            best_clf = fitted_clf,avg_score,std
    return best_clf
```


```python
best_clf = chooseBest()
```

    Evaluating ExtraTreesRegressor
    Score is: 0.622780015218.
    Score is: 0.620322556033.
    Score is: 0.627694933586.
    ExtraTreesRegressor scored: 0.623599168279 with std: 0.00306499010315
    ExtraTreesRegressor is currently the best
    


```python
best_clf
```




    (Pipeline(steps=[('extract_features', FeatureUnion(n_jobs=1,
            transformer_list=[('QueryBagOfWords', Pipeline(steps=[('selector', ItemSelector(key='query_stemed')), ('vectorizer', CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
             dtype=<type 'numpy.int64'>, encoding=u'utf-8'...imators=100, n_jobs=-1, oob_score=False, random_state=None,
               verbose=0, warm_start=False))]),
     0.6235991682790113,
     0.0030649901031484347)



## export output


```python
best_clf[0].fit(train,train["median_relevance"])
print "Done training"
```

    Done training
    


```python
with open("clf.pkl",'wb') as f:
    pickle.dump(best_clf,f)
```


```python
gc.collect()
```




    66




```python
test = preprocess(test)
print "Done text Preprocess"
```

    Done cleaning HTML
    Done removing stopwords
    Done stemming
    Done text Preprocess
    


```python
test = extract_features(test)
print "Done text Feature Extraction"
```

    Done counting lenghts
    Done counting stemed lenghts
    Done calculate length differences
    Done calc change ratio
    Done calc length ratio
    Done calc length ratio
    Done flaging empty description
    Done calc BM25
    Done calc similar words
    Done text Feature Extraction
    


```python
list(test.columns)
```




    ['id',
     'query',
     'product_title',
     'product_description',
     'product_description_clean',
     'product_title_clean',
     'query_clean',
     'product_description_stemed',
     'product_title_stemed',
     'query_stemed',
     'query_init_len',
     'desc_init_len',
     'title_init_len',
     'query_stemed_len',
     'product_description_stemed_len',
     'product_title_stemed_len',
     'query_diff_len',
     'desc_diff_len',
     'title_diff_len',
     'query_change_ratio',
     'desc_change_ratio',
     'title_change_ratio',
     'query_title_ratio',
     'query_desc_ratio',
     'title_desc_ratio',
     'no_desc',
     'BM25Title',
     'BM25Description',
     'query_title_similar_1gram_len',
     'query_desc_similar_1gram_len',
     'title_desc_similar_1gram_len',
     'query_title_similar_1gram_percent',
     'query_desc_similar_1gram_percent',
     'title_desc_similar_1gram_percent',
     'query_title_diff_1gram',
     'query_desc_diff_1gram',
     'title_desc_diff_1gram',
     'title_query_diff_1gram',
     'desc_query_diff_1gram',
     'desc_title_diff_1gram',
     'query_title_sym_diff_1gram',
     'query_desc_sym_diff_1gram',
     'title_desc_sym_diff_1gram',
     'query_title_similar_2gram_len',
     'query_desc_similar_2gram_len',
     'title_desc_similar_2gram_len',
     'query_title_similar_2gram_percent',
     'query_desc_similar_2gram_percent',
     'title_desc_similar_2gram_percent',
     'query_title_diff_2gram',
     'query_desc_diff_2gram',
     'title_desc_diff_2gram',
     'title_query_diff_2gram',
     'desc_query_diff_2gram',
     'desc_title_diff_2gram',
     'query_title_sym_diff_2gram',
     'query_desc_sym_diff_2gram',
     'title_desc_sym_diff_2gram',
     'query_title_similar_3gram_len',
     'query_desc_similar_3gram_len',
     'title_desc_similar_3gram_len',
     'query_title_similar_3gram_percent',
     'query_desc_similar_3gram_percent',
     'title_desc_similar_3gram_percent',
     'query_title_diff_3gram',
     'query_desc_diff_3gram',
     'title_desc_diff_3gram',
     'title_query_diff_3gram',
     'desc_query_diff_3gram',
     'desc_title_diff_3gram',
     'query_title_sym_diff_3gram',
     'query_desc_sym_diff_3gram',
     'title_desc_sym_diff_3gram']




```python
with open("clf.pkl",'rb') as f:
    best_clf = pickle.load(f)
```


```python
best_clf
```




    (Pipeline(steps=[('extract_features', FeatureUnion(n_jobs=1,
            transformer_list=[('QueryBagOfWords', Pipeline(steps=[('selector', ItemSelector(key='query_stemed')), ('vectorizer', CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict',
             dtype=<type 'numpy.int64'>, encoding=u'utf-8'...imators=100, n_jobs=-1, oob_score=False, random_state=None,
               verbose=0, warm_start=False))]),
     0.6235991682790113,
     0.0030649901031484347)




```python
test.shape
```




    (22513, 73)




```python
total_submission = pd.DataFrame()
BATCH = 500
i=0
while i*BATCH<test.shape[0]:
    predictions = [pred for pred in best_clf[0].predict(test[:][BATCH*i:BATCH*(i+1)])]
    cfd_dest, round_dest, ceil_dest, floor_dest = prediction_distributions(predictions)
    cfd_dest = [int(num) for num in cfd_dest]
    submission = pd.DataFrame({"id": test[:][BATCH*i:BATCH*(i+1)]["id"], "prediction": cfd_dest})
    total_submission = pd.concat([total_submission,submission])
    i += 1
print total_submission.shape
```

    (22513, 2)
    


```python
i
```




    46




```python
total_submission.to_csv("res.csv", index=False)
```


```python
test.describe()
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>query_init_len</th>
      <th>desc_init_len</th>
      <th>title_init_len</th>
      <th>query_stemed_len</th>
      <th>product_description_stemed_len</th>
      <th>product_title_stemed_len</th>
      <th>query_diff_len</th>
      <th>desc_diff_len</th>
      <th>title_diff_len</th>
      <th>...</th>
      <th>title_desc_similar_3gram_percent</th>
      <th>query_title_diff_3gram</th>
      <th>query_desc_diff_3gram</th>
      <th>title_desc_diff_3gram</th>
      <th>title_query_diff_3gram</th>
      <th>desc_query_diff_3gram</th>
      <th>desc_title_diff_3gram</th>
      <th>query_title_sym_diff_3gram</th>
      <th>query_desc_sym_diff_3gram</th>
      <th>title_desc_sym_diff_3gram</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>...</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
      <td>22513.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16328.282992</td>
      <td>2.356505</td>
      <td>65.717363</td>
      <td>8.443477</td>
      <td>2.296362</td>
      <td>49.224359</td>
      <td>8.150402</td>
      <td>0.060143</td>
      <td>16.493004</td>
      <td>0.293075</td>
      <td>...</td>
      <td>0.175269</td>
      <td>1.955937</td>
      <td>2.161418</td>
      <td>29.710212</td>
      <td>36.020433</td>
      <td>265.157065</td>
      <td>258.641363</td>
      <td>37.976369</td>
      <td>267.318483</td>
      <td>288.351575</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9424.576451</td>
      <td>0.845015</td>
      <td>118.010674</td>
      <td>3.267144</td>
      <td>0.775212</td>
      <td>99.341340</td>
      <td>3.040978</td>
      <td>0.261427</td>
      <td>26.046574</td>
      <td>0.610858</td>
      <td>...</td>
      <td>0.446344</td>
      <td>3.569497</td>
      <td>3.740122</td>
      <td>19.826356</td>
      <td>17.896238</td>
      <td>455.232832</td>
      <td>452.748355</td>
      <td>18.809260</td>
      <td>455.214390</td>
      <td>452.857089</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8201.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>12.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>63.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>16329.000000</td>
      <td>2.000000</td>
      <td>40.000000</td>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>29.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>30.000000</td>
      <td>36.000000</td>
      <td>162.000000</td>
      <td>156.000000</td>
      <td>36.000000</td>
      <td>162.000000</td>
      <td>186.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>24464.000000</td>
      <td>3.000000</td>
      <td>85.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>62.000000</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.181818</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>42.000000</td>
      <td>48.000000</td>
      <td>348.000000</td>
      <td>336.000000</td>
      <td>48.000000</td>
      <td>348.000000</td>
      <td>366.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>32671.000000</td>
      <td>6.000000</td>
      <td>2984.000000</td>
      <td>42.000000</td>
      <td>6.000000</td>
      <td>2658.000000</td>
      <td>38.000000</td>
      <td>2.000000</td>
      <td>1097.000000</td>
      <td>10.000000</td>
      <td>...</td>
      <td>5.400000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>186.000000</td>
      <td>204.000000</td>
      <td>11671.000000</td>
      <td>11671.000000</td>
      <td>204.000000</td>
      <td>11677.000000</td>
      <td>11725.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 64 columns</p>
</div>



![alt text](Classifier-Results.png "Classifier uploaded Results.")
