from sentence_transformers import SentenceTransformer, util

class SentenceSimilarityCalculator:
    def __init__(self):
        self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')

    def calculate_similarity(self, sentences1, sentences2):
        embeddings1 = self.model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences2, convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        return similarity_scores
    
    def encode(self, sentences):
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings


if __name__ == "__main__":
    sentences1 = ['The cat sits outside', 'My name is John']
    sentences2 = ['The dog plays in the garden', 'the name is Sam']

    sentence_similarity_calculator = SentenceSimilarityCalculator()
    similarity_scores = sentence_similarity_calculator.calculate_similarity(sentences1, sentences2)
    print(similarity_scores)

