from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from laserembeddings import Laser


class BertSentenceConverter:
    def __init__(self, model_name='all-MiniLM-L6-v2', device_number=0):
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def encode_to_vec(self, sentences, token=None, nlp=False):
        if type(sentences) == str:
            sentences = [sentences]

        for sentence in sentences:
            if len(sentence) > 0 and sentence[-1] != ".":
                sentence += "."

        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        return embeddings.detach().numpy().tolist()


class LaserSentenceConverter:

    def __init__(self):
        self.laser = Laser()

    def encode_to_vec(self, sentences, languages):
        if type(sentences) == str:
            sentences = [sentences]

        for sentence in sentences:
            if len(sentence) > 0 and sentence[-1] != ".":
                sentence += "."

        embeddings = self.laser.embed_sentences(sentences, lang=languages)

        return embeddings
