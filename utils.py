import pymorphy2
import nltk
from scipy.spatial import distance
import gensim
import numpy as np
from navec import Navec
import os
import random
import requests
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('punkt')

path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
if not os.path.exists(path):
    os.system('wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar')

navec = Navec.load(path)
morph = pymorphy2.MorphAnalyzer()

categories = np.array(["животные", "музыка", "спорт", "литература"])
categories_eng = np.array(["animals", "music", "sport", "literature"])


class KeywordsLoader:
    def __init__(self, keywords_dir):
        self.nouns = [[] for i in range(4)]
        self.nouns_actors = [[] for i in range(4)]
        self.keywords_dir = keywords_dir
        morph = pymorphy2.MorphAnalyzer()
        with open(keywords_dir + '/keywords_tasks.txt', 'r') as f:
            words = f.read()
            words = [morph.parse(token.lower())[0].normal_form for token in words.split('\n')]
            self.keywords_tasks = set(words)

        self.load_keyword_nouns()
        self.load_keyword_actors()

    def load_keyword_nouns(self):
        for i, cat in enumerate(categories_eng):
            with open(self.keywords_dir + f'/nouns/true_keywords_nouns_{cat}.txt', 'r') as f:
                words = f.read()
            for el in words.split('\n')[:-1]:
                if not el in self.keywords_tasks:
                    self.nouns[i].append(el)

            with open(self.keywords_dir + f'/wow_keywords/wow_keywords_{cat}.txt', 'r') as f:
                words = f.read()
            for el in words.split('\n')[:-1]:
                if not el in self.keywords_tasks and 'NOUN' in morph.parse(el)[0].tag:
                    self.nouns[i].append(el)
        return self.nouns

    def load_keyword_actors(self):
        for i, cat in enumerate(categories_eng):
            with open(self.keywords_dir + f'/actors/true_keywords_nouns_actors_{cat}.txt', 'r') as f:
                words = f.read()

            for el in words.split('\n')[:-1]:
                if not el in self.keywords_tasks:
                    self.nouns_actors[i].append(el)
        return self.nouns_actors


class MaskCreator:
    def __init__(self, bigram_model, nouns, nouns_actors):
        self.morph = pymorphy2.MorphAnalyzer()
        self.tokenizer = nltk.WordPunctTokenizer()
        self.bigram_mod = gensim.models.Phrases.load(bigram_model)
        self.nouns = nouns
        self.nouns_actors = nouns_actors

    def make_bigrams(self, doc):
        return self.bigram_mod[doc]

    def mask(self, text, category=4):
        masks_dict = []
        tokens = self.tokenizer.tokenize(text.lower())
        tokens_normal = [self.morph.parse(w)[0].normal_form for w in tokens]
        tokens_bigrammed = self.make_bigrams(tokens_normal)

        if len(tokens_bigrammed) < len(tokens):
            ind_go = 0
            for i in range(len(tokens_bigrammed)):
                if tokens_normal[ind_go] != tokens_bigrammed[i]:
                    tokens = tokens[:ind_go] + [tokens_bigrammed[i]] + tokens[ind_go + 2:]
                    ind_go += 2
                else:
                    ind_go += 1

        if category == 4:
            now_keywords = self.nouns[0] + self.nouns[1] + self.nouns[2] + self.nouns[3]
        else:
            now_keywords = self.nouns[category] + self.nouns_actors[category]

        prev_words = []
        for ind, token in enumerate(tokens):
            word = self.morph.parse(token.lower())[0].normal_form
            if word in now_keywords:
                if word not in masks_dict:
                    masks_dict.append(word)
                prev_words.append(tokens[ind])
                tokens[ind] = 'mask' + str(masks_dict.index(word, 0))
        text = nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(tokens)
        return text, masks_dict


def find_best(ideal_dist, vocab, new_masks, top_k=1):
    top = []
    for i in range(top_k):
        min_val = 1
        min_new_word = ''
        for new_word in vocab:
            if not new_word in navec \
                    or new_word in new_masks \
                    or new_word in top:
                continue
            now_dist = distance.cosine(navec[new_word], ideal_dist)
            if now_dist < min_val:
                min_val = now_dist
                min_new_word = new_word
        top.append(min_new_word)
    return top


def convert(mask_creator, keywords, sentence, category_from, category_to, first_clever, diffrent_category):
    sentence_with_masks, masks = mask_creator.mask(sentence)
    # print(sentence_with_masks)
    new_masks = ['' for i in range(len(masks))]
    words_in_sentence = word_tokenize(sentence_with_masks)
    get_first_mask = False
    for word in words_in_sentence:
        if word[:4] == 'mask' and masks[int(word[4:])] in keywords.nouns_actors[category_from]:
            first_mask = navec[masks[int(word[4:])]]
            if first_clever:
                top = find_best(first_mask + navec[categories[category_to]] - navec[categories[category_from]],
                                keywords.nouns_actors[category_to], [], 4)
                new_word = top[random.randint(0, 3)]
            else:
                new_word = keywords.nouns_actors[category_to][random.randint(0, 5)]

            new_first_mask = navec[new_word]
            new_masks[int(word[4:])] = new_word
            get_first_mask = True
            break

    vocab_nouns = keywords.nouns[category_to]

    for ind, word in enumerate(words_in_sentence):
        if word[:4] == 'mask':
            if not get_first_mask:
                first_mask = navec[masks[int(word[4:])]]
                new_word = keywords.nouns[category_to][random.randint(0, 19)]
                new_first_mask = navec[new_word]
                new_masks[int(word[4:])] = new_word
                get_first_mask = True
            elif new_masks[int(word[4:])] == '':
                if navec.get(masks[int(word[4:])]) is None:
                    new_masks[int(word[4:])] = masks[int(word[4:])]
                    # print("None in navic")
                    continue

                if diffrent_category:
                    ideal_dist = navec[masks[int(word[4:])]] - navec[categories[category_from]] + navec[
                        categories[category_to]]
                else:
                    ideal_dist = new_first_mask - first_mask + navec[masks[int(word[4:])]]

                new_masks[int(word[4:])] = find_best(ideal_dist=ideal_dist,
                                                     vocab=vocab_nouns,
                                                     new_masks=new_masks)[0]
            words_in_sentence[ind] = new_masks[int(word[4:])]

    return ' '.join(words_in_sentence), new_masks
