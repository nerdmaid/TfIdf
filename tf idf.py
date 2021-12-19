from math import log

class CountVectorizer:
    """Convert a collection of text documents to a matrix of token counts.
     Input  is expected to be a sequence of strings"""

    def __init__(self, lowercase=True):
        self.lowercase = lowercase

    def get_keys(self, corpus):
        if self.lowercase:
            for_keys = list(set((' '.join(corpus).lower()).split()))
        else:
            for_keys = list(set((' '.join(corpus)).split()))
        return for_keys

    def fit_transform(self, corpus):
        """Learn the vocabulary dictionary and return document-term matrix"""
        vectorized_counter = []
        for_keys = CountVectorizer.get_keys(self, corpus)
        empty_counter = dict.fromkeys(for_keys, 0)
        for text in corpus:
            if self.lowercase:
                list_of_words = text.lower().split()
            else:
                list_of_words = text.split()
            for word in list_of_words:
                empty_counter[word] += 1
            vectorized_counter.append(list(empty_counter.values()))
            empty_counter = dict.fromkeys(for_keys, 0)
        return vectorized_counter

    def get_feature_names(self, corpus):
        """Get output feature names for transformation"""
        for_keys = CountVectorizer.get_keys(self, corpus)
        return for_keys

class TfidfTransformer:
    def tf_transform(self, matrix):
        new_matrix = []
        for i in range(len(matrix)):
            new_matrix.append([])
            for j in range(len(matrix[i])):
                new_matrix[i].append(round(matrix[i][j] / sum(matrix[i]), 3))
        return new_matrix

    def idf_transform(self, matrix):
        idf = []
        for j in range(len(matrix[0])):
            idf.append(
                round(log((len(matrix) + 1) / (sum([(matrix[i][j] != 0) for i in range(len(matrix))]) + 1)) + 1, 2))
        return idf

    def fit_transform(self, matrix):
        tf_transformed = self.tf_transform(matrix)
        idf_transformed = self.idf_transform(matrix)
        for i in range(len(tf_transformed)):
            for j in range(len(tf_transformed[i])):
                tf_transformed[i][j] = round(tf_transformed[i][j] * idf_transformed[j], 3)
        return tf_transformed

class TfidfVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self._tf_idf_transformer = TfidfTransformer()

    def fit_transform(self, corpus):
        matrix = super().fit_transform(corpus)
        tf_idf_matrix = self._tf_idf_transformer.fit_transform(matrix)
        return tf_idf_matrix


if __name__ == '__main__':
    corpus = [
     'Crock Pot Pasta Never boil pasta again',
     'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names(corpus))

    print(tfidf_matrix)