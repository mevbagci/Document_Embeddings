import wikipedia
import os
import spacy
from process import *
from BERT_converter import *
from langdetect import detect
from typing import Union
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
doc_embs = './doc_embs/'


def extract_wikipage(name: str):
    try:
        wikipage = wikipedia.page(name)
        text = wikipage.content
        return text
    except Exception as ex:
        print(ex)
        return f"Failed for {name}"


def spacy_sentence(text: str):
    doc = nlp(text)
    sentences = []
    lang = []
    for c, sent in enumerate(doc.sents):
        sentences.append(sent.text)
        try:
            lang.append(detect(sent.text))
        except:
            lang.append("en")
    return sentences, lang


def get_feature_vecs(model_type, text):
    doc_embeddings = Sentence_Configurations()
    doc_embeddings.__init__()
    sentences, lang = spacy_sentence(text)
    sentences = sentences[:20]
    lang = lang[:20]
    add_documents(0, sentences, doc_embeddings)
    if isinstance(model_type, LaserSentenceConverter):
        embeddings = model_type.encode_to_vec(sentences, lang)
    else:
        embeddings = model_type.encode_to_vec(sentences)
    doc_embeddings.docs[0] = sentences
    doc_embeddings.docs[0] = embeddings
    doc_embeddings.run_all(0)
    for emb_type in tqdm(['full', 'top', 'bottom', 'pert',
                          'tf_pert',
                          # 'tf_idf_2_4',
                          # 'tf_idf_4_4',
                          'attn_pert',
                          'attn_tf_pert']):
        doc_dir = doc_embs
        print(doc_embeddings.docs[0])
        print(doc_embeddings.doc_vecs[0])
        with open("{}doc_{}_{}.txt".format(doc_dir, 0, emb_type), 'w+') as f:
            for j in range(len(doc_embeddings.doc_vecs[0][emb_type])):
                f.write("{}".format(doc_embeddings.doc_vecs[0][emb_type][j]))
                f.write("\n")
        f.close()


def add_documents(i, sentences, doc_embeddings):
    doc_embeddings.doc_ids.append(i)
    doc_embeddings.doc_texts[i] = sentences


if __name__ == '__main__':
    article = extract_wikipage("Death")
    # sen, langs = spacy_sentence(article)
    laser_sen = LaserSentenceConverter()
    get_feature_vecs(laser_sen, article)
