import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import preprocessFacebook


def vectorize(filename):
    dataset = preprocessFacebook.preprocess(filename)
    dataset = dataset[pd.notnull(dataset['Description'])]
    dataset.columns = ['Description', 'Label']
    dataset['category_id'] = dataset['Label'].factorize()[0]
    dataset.head()

    ################################### Vectorization #####################################

    tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1', ngram_range=(1, 2), lowercase=False, min_df=5)
    features_d = tfidf.fit_transform(dataset.Description).toarray()
    features_d.shape
    return dataset
################################### Train -Test Split and Transformation #####################################

def trainData(filename):
    dataset = vectorize(filename)
    X_train, X_test, y_train, y_test = train_test_split(dataset['Description'], dataset['Label'], random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_counts = count_vect.transform(X_test)
    y_te = tfidf_transformer.transform(X_test_counts)

    clf_SVM_sig = svm.SVC(kernel='sigmoid').fit(X_train_tfidf, y_train)
    y_pred_SVM_sig = clf_SVM_sig.predict(y_te)
    return [clf_SVM_sig, y_pred_SVM_sig, count_vect,y_test]


def measureModelPerformance(filename):
    model = trainData(filename)
    y_pred_SVM_sig = model[1]
    y_test = model[3]
    print ('SVM - Sigmoid Accuracy:', accuracy_score(y_test, y_pred_SVM_sig))
    print ('SVM - Sigmoid F1 score:', f1_score(y_test, y_pred_SVM_sig, average='weighted'))
    print ('SVM - Sigmoid Recall:', recall_score(y_test, y_pred_SVM_sig, average='weighted'))
    print ('SVM - Sigmoid Precision:', precision_score(y_test, y_pred_SVM_sig, average='weighted'))

measureModelPerformance("trainingData.csv")