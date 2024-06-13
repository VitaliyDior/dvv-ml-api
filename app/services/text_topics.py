import operator
from typing import List, Dict

from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Doc2Vec
from nltk.tokenize import sent_tokenize
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, \
    strip_numeric, remove_stopwords
import numpy as np
import itertools


class TextTopicsService:
    d2v_model: Doc2Vec

    def __init__(self, app: FastAPI):
        self.d2v_model = app.state.d2v_model

    def split_to_topics(self, text: str) -> List[List[str]]:
        docs, sents = self.preprocess_tokenize(text)
        vectors = self.tokens_to_vectors(docs)
        distances = self.calculate_distances(vectors)
        groups = self.group_distances(distances)
        sents_groups = []
        for group in groups:
            sent_group = []
            for sent_index in group:
                sent_group.append(sents[sent_index])
            sents_groups.append(sent_group)
        return sents_groups

    def preprocess_tokenize(self, text: str) -> tuple[List[List[str]], List[str]]:
        sents = sent_tokenize(text)
        docs = []
        for sent in sents:
            tokens = preprocess_string(str(sent).lower(),
                                       filters=[strip_tags, strip_punctuation, strip_multiple_whitespaces,
                                                strip_numeric, remove_stopwords])
            docs.append(tokens)
        return docs, sents

    def tokens_to_vectors(self, doc: List[List[str]]) -> np.ndarray:
        return np.array([self.d2v_model.infer_vector(sent, alpha=0.025, epochs=300) for sent in doc])

    def calculate_distances(self, vectors: np.ndarray) -> List[List[float]]:
        distanses = []

        for vector in vectors:
            sent_dists = []
            for y in range(len(vectors)):
                d = cosine_similarity(vector.reshape(1, -1), vectors[y].reshape(1, -1))
                sent_dists.append(d[0][0])
            distanses.append(sent_dists)
        return distanses

    def group_distances(self, distances: List[List[float]] ) -> List[List[int]]:
        distance_matrix = np.round(np.matrix(distances), decimals=2)
        deg_matrix = np.degrees(np.arccos(distance_matrix))
        similarity_dict = {}
        for sent_index, sent_degress in enumerate(deg_matrix.tolist()):
            similarity_dict[sent_index] = []
            for col_index, relative_deg in enumerate(sent_degress):
                if relative_deg == 0:
                    continue
                if relative_deg > 45:
                    continue
                similarity_dict[sent_index].append(col_index)
        return self.filter_unique_groups(self.make_groups(similarity_dict))

    def make_groups(self, similarity_dict: Dict[int, List[int]]) -> List[List[int]]:
        groups_list = []
        for sent_index in similarity_dict.keys():
            group = [sent_index]
            group.extend(similarity_dict[sent_index])
            group.sort()
            groups_list.append(group)
        return groups_list

    def filter_unique_groups(self, groups_list: List[List[int]]) -> List[List[int]]:
        groups_list.sort()
        return list(groups_list for groups_list, _ in itertools.groupby(groups_list))
