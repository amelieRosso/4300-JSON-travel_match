import math
import os
import re
import json
import numpy as np
import nltk
nltk.data.path
from nltk.corpus import stopwords
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from collections import Counter

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
    

    # remove URLs in the text
    pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
    text = re.sub(pattern, "", text)

    # create basic tokens
    tokens = nltk.word_tokenize(text)

    # get list of stopwords in English (multilingual processing seems to be much harder and most of the text is english except for place names)
    stop_words = set(stopwords.words("english"))

    custom_stopwords = {
        "site", "location", "place", "area", "go", "visit",  
    }

    
    all_stopwords = stop_words.union(custom_stopwords)
    # remove stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in all_stopwords]  

    # if lemmatizer takes too long, switch to using stemming
    try:
      lemmatizer = nltk.stem.WordNetLemmatizer()
      lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    except Exception as e:
      print("Lemmatizer failed:", e)
      stemmer = nltk.stem.PorterStemmer()
      lemmatized_tokens = [stemmer.stem(token) for token in filtered_tokens]

    return np.array(lemmatized_tokens)

"""
Expects: [data] to be the data from whc_sites_2021 in the json file. 
Outputs: a dict of site_id to tokenized descriptions.
"""
# create dict of doc_id to tokenized description
def create_tokenized_dict(filtered_data: List[dict]) -> dict:
    tokenized_dict = {}
    for site_id, site in enumerate(filtered_data):
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

svd = TruncatedSVD(n_components=200)
vectorizer = TfidfVectorizer()

tokenized_dict = create_tokenized_dict(data)
docs = [" ".join(tokenized_dict[site_id]) for site_id in sorted(tokenized_dict.keys()) ]
vectorizer_no_filter = TfidfVectorizer()
docs_tfidf = vectorizer_no_filter.fit_transform(docs)
svd_no_filter = TruncatedSVD(n_components=200)
reduced_docs = svd_no_filter.fit_transform(docs_tfidf)

def get_reduced_docs(filtered_data):
  # vectorizer = TfidfVectorizer()
  # svd = TruncatedSVD(n_components = 75)
  filtered_tokenized_dict = create_tokenized_dict(filtered_data)
  filtered_docs = [" ".join(filtered_tokenized_dict[site_id]) for site_id in sorted(filtered_tokenized_dict.keys()) ]
  filtered_docs_tfidf = vectorizer.fit_transform(filtered_docs)
  return svd.fit_transform(filtered_docs_tfidf), vectorizer, svd

def extract_svd_dimension_terms():
  
  terms = vectorizer_no_filter.get_feature_names_out()
  
  # Create the output text file
  output_file = "svd_75_dimension_terms.txt"
  
  with open(output_file, 'w', encoding='utf-8') as f:
      for dim_idx in range(75):
          # Get the component vector for this dimension
          component = svd_no_filter.components_[dim_idx]
          
          # Get the indices of all terms sorted by importance
          sorted_indices = component.argsort()[::-1][:100]
          
          # Write the dimension header
          f.write(f"Dimension {dim_idx}:\n")
          
          list=[]
          for idx in sorted_indices:
              term = terms[idx]
              list.append(term)
          f.write(f"{list}")
          
          f.write("\n")
  
  print(f"Dimension terms saved to {os.path.abspath(output_file)}")
  return output_file


def extract_top5terms(vectorizer, svd, output_path="top5(75dim)_terms.py"):
    terms = vectorizer.get_feature_names_out()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Auto-generated SVD dimension labels\n")
        f.write("labels = [\n")
        for dim_idx in range(svd.components_.shape[0]):
            component = svd.components_[dim_idx]
            sorted_indices = component.argsort()[::-1][:5]
            top_terms = [f'"{terms[i]}"' for i in sorted_indices]
            f.write(f"    [{', '.join(top_terms)}],\n")
        f.write("]\n")

    print(f"Saved Python labels module to {os.path.abspath(output_path)}")
    

#create reduced_docs (global var)

# uncomment it to play with the dimension
# # Step 3: Get term-topic matrix (terms per dimension)
# terms = vectorizer.get_feature_names_out()
# term_topic_matrix = svd.components_  

# # Step 4: For each of the top 10 dimensions, get top 10 terms
# for dim in range(10):
#     top_indices = term_topic_matrix[dim].argsort()[::-1][:10]  
#     top_words = [terms[i] for i in top_indices]
#     print(f"Dimension {dim + 1}: {top_words}")


def transform_query_to_svd(query: str, vectorizer, svd, weights: dict = None):
    if weights is None:
        weights = {}
    query_tokens = preprocess_description(query)
    counter = Counter(query_tokens)

    # Apply weights based on token frequency 
    weighted_tokens = []
    for token, count in counter.items():
        weight = count  # weight = frequency count => tf = w/|query|
        weighted_tokens.extend([token] * weight)
    # query_str = " ".join(query_tokens)
    # Join the weighted tokens back into a string
    query_str = " ".join(weighted_tokens)
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
  return [(sim[i],i) for i in ids[:9]]


# def extract_svd_tags(reduced_query, reduced_docs, svd, vectorizer doc_text=""):
#    # Project scores back into term space
#     query_term_scores = np.dot(reduced_query, svd.components_).flatten()
#     doc_term_scores = np.dot(reduced_docs, svd.components_).flatten()

#     # Match relevance by multiplying query and doc term scores
#     match = query_term_scores * doc_term_scores

#     terms = vectorizer.get_feature_names_out()
#     doc_text_lower = doc_text.lower()
#     tags = []

#     top_indices = match.argsort()[::-1]

#     for i in top_indices:
#         if len(tags) >= 5:
#             break
#         term = terms[i]
#         if term in doc_text_lower:
#             tags.append(term)

#     return tags

# This is causing errors (specifically line 156)
def extract_svd_tags(reduced_query, reduced_doc, svd, vectorizer, doc_text=""):
    
    dim_scores = reduced_query.flatten() * reduced_doc.flatten()
    top_dims = dim_scores.argsort()[::-1]  

    terms = vectorizer.get_feature_names_out()
    doc_text_lower = preprocess_description(doc_text)

    tags = []
    seen_roots = set()

    for dim in top_dims:
        if len(tags) >= 5:
            break

        # Get terms associated with this dimension
        component = svd.components_[dim]
        top_term_indices = component.argsort()[::-1]

        for idx in top_term_indices:
            term = terms[idx]
            if term in doc_text_lower:
                root = term.lower()
                if root not in seen_roots:
                    tags.append(term)
                    seen_roots.add(root)
                    break  # go to next dimension

    return tags 

"""
Expects: [data] from top 10 sites be the data from whc_sites_2021 in the json file. 
Outputs: a dict of site_id to tokenized descriptions.
"""
# create dict of doc_id to tokenized description
def create_tokenized_dict_10(query: str, reduced_docs, filtered_data, vectorizer, svd) -> dict:
    
    reduced_query, _ = transform_query_to_svd(query, vectorizer, svd)
    similarity_score_to_site_index_tuple = svd_index_search(reduced_query, reduced_docs)

    site_info_dict = {}
    for sim_score, site_index in similarity_score_to_site_index_tuple:
       site_info_dict[site_index] = filtered_data[site_index]

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
def build_inverted_index(query: str, reduced_docs, filtered_data, vectorizer, svd) -> dict:
    
    tokenized_dict_10 = create_tokenized_dict_10(query, reduced_docs, filtered_data, vectorizer, svd)

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
def compute_idf(query: str, n_sites, reduced_docs, filtered_data, vectorizer, svd, min_df=0, max_df_ratio=0.95):
    inv_idx = build_inverted_index(query, reduced_docs, filtered_data, vectorizer, svd)

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
def compute_doc_norms(query: str, n_sites, reduced_docs, filtered_data, vectorizer, svd):
    inv_idx = build_inverted_index(query, reduced_docs, filtered_data, vectorizer, svd)
    idf = compute_idf(query, n_sites, reduced_docs, filtered_data, vectorizer, svd)

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
def accumulate_dot_scores(query_word_counts: dict, query: str, reduced_docs, n_sites, filtered_data, vectorizer, svd) -> dict:
    inv_idx = build_inverted_index(query, reduced_docs, filtered_data, vectorizer, svd)
    idf = compute_idf(query, n_sites, reduced_docs, filtered_data, vectorizer, svd)
    dot_scores_dict = {}
    for word, query_tf in query_word_counts.items():
      if word in idf and word in inv_idx:
        idf_value = idf[word]
        for site_id, tf in inv_idx[word]:
          numer = (query_tf * idf_value) * (tf * idf_value) 
          if site_id not in dot_scores_dict:
            dot_scores_dict[site_id] = 0
          dot_scores_dict[site_id] += numer

    tokenized_dict_10 = create_tokenized_dict_10(query, reduced_docs, filtered_data, vectorizer, svd)
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
    filtered_reduced_docs,
    vectorizer,
    svd,
    score_func=accumulate_dot_scores,
    filtered_data = data
) -> List[Tuple[int, int]]:
    len_sites = len(filtered_data)
    if filtered_data == data:
      idf = compute_idf(query, len_sites, reduced_docs, data, vectorizer_no_filter, svd_no_filter)
      doc_norms = compute_doc_norms(query, len_sites, reduced_docs, data, vectorizer_no_filter, svd_no_filter)
      
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

      numer = score_func(query_word_counts_dict, query, reduced_docs, len_sites, data, vectorizer_no_filter, svd_no_filter)
      reduced_query, query_tfidf = transform_query_to_svd(query, vectorizer_no_filter, svd_no_filter)
      similarity_score_to_site_index_tuple = svd_index_search(reduced_query, reduced_docs)
      similarity_score_to_site_index_dict = {t[1]: t[0] for t in similarity_score_to_site_index_tuple}
      for site_id, score in numer.items():
        index_search_list_tuples.append(((score / ((doc_norms[site_id] * abs_q) + 1)), site_id, similarity_score_to_site_index_dict[site_id]))

      return sorted(index_search_list_tuples, key=lambda x: (x[0], x[2]), reverse=True)

    else:
      idf = compute_idf(query, len_sites, filtered_reduced_docs, filtered_data, vectorizer, svd)
      doc_norms = compute_doc_norms(query, len_sites, filtered_reduced_docs, filtered_data, vectorizer, svd)

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

      numer = score_func(query_word_counts_dict, query, filtered_reduced_docs, len_sites, filtered_data, vectorizer, svd)
      reduced_query, query_tfidf = transform_query_to_svd(query, vectorizer, svd)
      similarity_score_to_site_index_tuple = svd_index_search(reduced_query, filtered_reduced_docs)
      similarity_score_to_site_index_dict = {t[1]: t[0] for t in similarity_score_to_site_index_tuple}
      for site_id, score in numer.items():
        index_search_list_tuples.append(((score / ((doc_norms[site_id] * abs_q) + 1)), site_id, similarity_score_to_site_index_dict[site_id]))
      
      return sorted(index_search_list_tuples, key = lambda x: (x[0], x[2]), reverse = True)


"""
Bert embeding
"""

bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def bert_search(
    query,
    filtered_data,
) -> List[Tuple[int, int]]:
  tokenized_dict = create_tokenized_dict(filtered_data)
  docs = [" ".join(tokenized_dict[site_id]) for site_id in sorted(tokenized_dict.keys()) ]
  query_tokens = preprocess_description(query)
  query_embedding = bert_model.encode(list(query_tokens))
  corpus_embeddings = bert_model.encode(docs)
  sim = cosine_similarity(corpus_embeddings, query_embedding).mean(axis=1)
  ids = sim.argsort()[::-1]
  return [(sim[i], i) for i in ids[:9]]

# bert tag not working well
# def extract_bert_tags(query, doc) -> List[str]:
  
#     query_tokens = preprocess_description(query)
#     doc_tokens = preprocess_description(doc)

#     # BERT encode token-level vectors
#     query_embeddings = bert_model.encode(list(query_tokens)) 
#     doc_embeddings = bert_model.encode(list(doc_tokens))

#     # Compute similarity matrix and average similarity for each doc token
#     similarity_matrix = cosine_similarity(doc_embeddings, query_embeddings)  
#     max_similarities = similarity_matrix.max(axis=1)

#     # Sort tokens by descending max similarity
#     top_indices = max_similarities.argsort()[::-1]

#     seen = set()
#     tags = []
#     for idx in top_indices:
#         token = doc_tokens[idx]
#         if token not in seen and token in doc_tokens:
#             tags.append(token)
#             seen.add(token)
#         if len(tags) == 5:
#             break
#     return tags



