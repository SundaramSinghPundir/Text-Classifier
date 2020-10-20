# Text-Classifier
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

emails = pd.read_csv('.csv')
print emails.shape(10000, 3)

def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    
keys_to_extract = ['Thank Youfor applying']
 for line in lines:
      if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email

def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'), 
        'to': map_to_list(emails, 'to'), 
        'from_': map_to_list(emails, 'from')
    }
    email_df = pd.DataFrame(parse_into_emails(emails.message))
    mail_df.drop(email_df.query("body == '' | to == '' | from_ == ''").index, inplace=True)
    
vect = TfidfVectorizer(stop_words='Thank You', max_df=0.50, min_df=2)
X = vect.fit_transform(email_df.body)
X_dense = X.todense()
crd = PCA(n_components=2).fit_transform(X_dense)
plt.scatter(crd[:, 0], crd[:, 1], c='m')
plt.show()

def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df
def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)
    features = vect.get_feature_names()
print top_feats_in_doc(X, features, 1, 10)

def top_mean_feats(X, features,grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()
        D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)
    n_clusters = 3

clf = KMeans(n_clusters=n_clusters, max_iter=100, init='k-means++', n_init=1)
labels = clf.fit_predict(X)

def top_feats_per_cluster(X, y, features, min_tfidf=0.1, top_n=25):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label) 
        feats_df = top_mean_feats(X, features, ids,    min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs
