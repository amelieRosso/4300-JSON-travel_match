import math
import os
import re
import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.corpus import stopwords
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# download tokenizer info 
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'whc_sites_2021_with_ratings.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

"""
Expects: [text] to be the "short description" field in the json file. Can be extended later to process the brief synthesis or other review text.
Outputs: a list of tokens as a numpy array for more efficient processing later.
"""
def preprocess_description(text:str) -> np.ndarray:
    #preprocessing steps followed from: https://spotintelligence.com/2022/12/21/nltk-preprocessing-pipeline/
    text = text.removeprefix("<p>").removesuffix("</p>")
    text = text.lower()
    #words_list = re.findall("[a-z]+", lowercase_text)
    

    # remove URLs in the text
    pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    text = re.sub(pattern, "", text)

    # create basic tokens
    tokens = nltk.word_tokenize(text)

    # get list of stopwords in English (multilingual processing seems to be much harder and most of the text is english except for place names)
    stop_words = stopwords.words("english")
    # remove stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]  

    # if lemmatizer takes too long, switch to using stemming
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return np.array(lemmatized_tokens)

# number of sites
n_sites = len(data)

"""
Expects: [data] to be the data from whc_sites_2021 in the json file. 
Outputs: a dict of site_id to tokenized descriptions.
"""
# create dict of doc_id to tokenized description
def create_tokenized_dict(data: List[dict]) -> dict:
    tokenized_dict = {}
    for site_id, site in enumerate(data):
      tokenize_description = preprocess_description(site['short_description'])
      tokenize_name = preprocess_description(site['Name'])

      # tokenize all reviews for a site
      reviews_text = ""
      for review in site['reviews']:
        reviews_text += review["text"] + " "
      tokenize_reviews = preprocess_description(reviews_text)

      # create a tokenized_dict to use in tf-idf matrix with site name, description, and review text
      tokenized_dict[site_id] = np.concatenate([tokenize_name, tokenize_description, tokenize_reviews])
    return tokenized_dict

tokenized_dict = create_tokenized_dict(data)

#create reduced_docs (global var)
docs = [" ".join(tokenized_dict[site_id]) for site_id in sorted(tokenized_dict.keys()) ]
vectorizer = TfidfVectorizer()
docs_tfidf = vectorizer.fit_transform(docs)
svd = TruncatedSVD(n_components=200)
reduced_docs = svd.fit_transform(docs_tfidf)

def transform_query_to_svd(query: str, vectorizer=vectorizer, svd=svd):
    query_tokens = preprocess_description(query)
    query_str = " ".join(query_tokens)
    query_tfidf = vectorizer.transform([query_str])
    reduced_query = svd.transform(query_tfidf)
    return reduced_query, query_tfidf


# SVD approach
def svd_index_search(
    reduced_query,
    reduced_docs,
) -> List[Tuple[int, int]]:

  sim = cosine_similarity(reduced_docs,reduced_query).flatten()
  ids = sim.argsort()[::-1]
  return [(sim[i],i) for i in ids[:10]]


def extract_svd_tags(reduced_query, reduced_docs, svd, vectorizer):
    # Project back into term space
    query_term_scores = np.dot(reduced_query, svd.components_).flatten()  
    doc_term_scores = np.dot(reduced_docs, svd.components_).flatten()
 
    match = query_term_scores * doc_term_scores

    # Get vocabulary
    terms = vectorizer.get_feature_names_out()

    # Find top N highest combined scores
    top_indices = match.argsort()[::-1][:5]
    tags = [terms[i] for i in top_indices if match[i] > 0]

    return tags

"""
Expects: [data] from top 10 sites be the data from whc_sites_2021 in the json file. 
Outputs: a dict of site_id to tokenized descriptions.
"""
# create dict of doc_id to tokenized description
def create_tokenized_dict_10(query: str) -> dict:
    
    reduced_query, query_tfidf = transform_query_to_svd(query, vectorizer, svd)
    similarity_score_to_site_index_tuple = svd_index_search(reduced_query, reduced_docs)

    site_info_dict = {}
    for sim_score, site_index in similarity_score_to_site_index_tuple:
       site_info_dict[site_index] = data[site_index]

    tokenized_dict_10 = {}
    for site_id, site in site_info_dict.items():
      tokenize_description = preprocess_description(site['short_description'])
      tokenize_name = preprocess_description(site['Name'])

      # create a tokenized_dict to use in tf-idf matrix with site name, description
      tokenized_dict_10[site_id] = np.concatenate([tokenize_name, tokenize_description])
    return tokenized_dict_10

"""
Expects: [tokenized_dict] a dict of site_id to tokenized descriptions. 
Outputs: a dict of tuples of term to (site id, tokenized descriptions).
"""
# create inverted index (list of tuples with smaller doc_ids appearing first)
def build_inverted_index(query: str) -> dict:
    
    tokenized_dict_10 = create_tokenized_dict_10(query)

    inverted_index_dict = {}

    for site_id, words in tokenized_dict_10.items():
      count_of_term_in_doc = {}
      
      for word in words:
        if word in count_of_term_in_doc:
          count_of_term_in_doc[word] += 1
        else:
          count_of_term_in_doc[word] = 1

      for word, count in count_of_term_in_doc.items():
        if word not in inverted_index_dict:
          inverted_index_dict[word] = []
        inverted_index_dict[word].append((site_id, count))

    for word_index in inverted_index_dict:
      inverted_index_dict[word_index].sort(key=lambda x: x[0])

    return inverted_index_dict

"""
Expects: [inv_idx, n_sites] inverted index from above and number of sites. 
Outputs: a dict of terms to idf vlaues.
"""
# create dict of term to idf value
def compute_idf(query: str, n_sites, min_df=0, max_df_ratio=0.95):
    inv_idx = build_inverted_index(query)

    idf_value_dict = {}

    for word, value_list in inv_idx.items():
      precent_in = len(value_list) / n_sites

      if len(value_list) >= min_df and precent_in <= max_df_ratio:
        idf_value = math.log2(n_sites/(1 + len(value_list)))
        idf_value_dict[word] = idf_value
    return idf_value_dict

"""
Expects: [inv_idx, idf, n_sites] inverted index from above, idf from above and number of sites. 
Outputs: an array of doc norms.
"""
def compute_doc_norms(query: str, n_sites):
    inv_idx = build_inverted_index(query)
    idf = compute_idf(query, n_sites)

    doc_norms_array = np.zeros(n_sites, dtype=float)
    for word, value_list in inv_idx.items():
      idf_value = idf.get(word, 0)
      for doc_id, tf in value_list:
        doc_norms_array[doc_id] += ((tf * idf_value) ** 2.0)
    doc_norms_array = np.sqrt(doc_norms_array)
    return doc_norms_array

"""
Expects: [query_word_counts, index, idf] dict of query words to tf values, inverted index from above, and idf from above. 
Outputs: dict of site ids to dot product value.
"""
# we need a query_word_counts (dict of words to tf of the query)
def accumulate_dot_scores(query_word_counts: dict, query: str) -> dict:
    inv_idx = build_inverted_index(query)
    idf = compute_idf(query, n_sites)
    dot_scores_dict = {}
    for word, query_tf in query_word_counts.items():
      if word in idf and word in inv_idx:
        idf_value = idf[word]
        for site_id, tf in inv_idx[word]:
          numer = (query_tf * idf_value) * (tf * idf_value) 
          if site_id not in dot_scores_dict:
            dot_scores_dict[site_id] = 0
          dot_scores_dict[site_id] += numer

    tokenized_dict_10 = create_tokenized_dict_10(query)
    for site_id in tokenized_dict_10:
      if site_id not in dot_scores_dict:
            dot_scores_dict[site_id] = 0
    print(f"dot_scores_dict {dot_scores_dict}")
    return dot_scores_dict

"""
Expects: [query, index, idf, doc_norms, score_func] query, inverted index from above, idf from above, doc norms form above, dot scores form above, and a tokenizer. 
Outputs: a list of tuples (cosine similarity value, site id)
"""
# need to figure out tokenizer
def index_search(
    query: str,
    score_func=accumulate_dot_scores,
    subset_indices=None
) -> List[Tuple[int, int]]:
    idf = compute_idf(query, n_sites)
    doc_norms = compute_doc_norms(query, n_sites)

    index_search_list_tuples = []

    tokenize_list = preprocess_description(query).tolist()
    query_word_counts_dict = {}

    for word in tokenize_list:
      query_word_counts_dict[word] = tokenize_list.count(word)
    
    q = 0
    for word in query_word_counts_dict:
      if word in idf:
        q += (query_word_counts_dict[word] * idf[word]) ** 2
    abs_q = math.sqrt(q)
    
    numer = score_func(query_word_counts_dict, query)
    reduced_query, query_tfidf = transform_query_to_svd(query, vectorizer, svd)
    similarity_score_to_site_index_tuple = svd_index_search(reduced_query, reduced_docs)
    similarity_score_to_site_index_dict = {t[1]: t[0] for t in similarity_score_to_site_index_tuple}
    for site_id, score in numer.items():
      if subset_indices is not None and site_id not in subset_indices:
        continue
      norm = doc_norms[site_id]
      svd_score = similarity_score_to_site_index_dict[site_id]
      cosine_sim = score/(norm * abs_q)
      index_search_list_tuples.append((cosine_sim, site_id, svd_score))
      # index = index_search_list_tuples.index(((score / (doc_norms[site_id] * abs_q)), site_id))

    #index_search_list_tuples.sort(reverse=True)[:10]
    return sorted(index_search_list_tuples, key=lambda x: (x[0], x[2]), reverse=True)
