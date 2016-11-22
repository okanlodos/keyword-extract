# -*- coding:utf-8 -*-
import math
from stopwords import stopwords
from textblob import TextBlob as tb


class tfidf():
    def __init__(self, lang, wordlist, urllist):
        self.wordlist = wordlist
        self.url_list = urllist
        self.single_List = []
        self.binary_List = []
        self.three_List = []
        self.Three_in_one_List = []
        self.SELECT = stopwords[lang]

    __tf = lambda self, word, blob: (blob.words.count(word) * 1.0) / ((len(blob.words)) * 1.0)

    __tf_2 = lambda self, word, blob: (blob.count(word) * 1.0) / ((len(blob)) * 1.0)

    __n_containing = lambda self, word, bloblist: sum(1 for blob in bloblist if word in blob.lower())

    __idf = lambda self, word, bloblist: math.log(
        ((len(bloblist)) * 1.0) / ((1 + self.__n_containing(word, bloblist)) * 1.0), 10)

    __tfidf = lambda self, word, blob, bloblist: self.__tf(word, blob) * self.__idf(word, bloblist)

    __tfidf_2 = lambda self, word, blob, bloblist: self.__tf_2(word, blob) * self.__idf(word, bloblist)

    def get_tfidf(self):
        bloblist = self.punctuation_clean()
        self.__single(bloblist)
        self.__binary(bloblist)
        self.__triple(bloblist)
        self.__keyword_show()

    def __stop_words(self, blob):
        new_blob = []
        try:
            for i in blob.words:
                if i not in self.SELECT:
                    new_blob.append(i)
        except:
            for i in blob.words:
                new_blob.append(i)
        return new_blob

    def __single(self, bloblist):

        for i, blob in enumerate(bloblist):
            blob = blob.lower()

            scores = {word: self.__tfidf(word, blob, bloblist) for word in self.__stop_words(blob)}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            self.single_List.append(sorted_words)

    def __binary(self, bloblist):
        bloblist2 = []
        for i, blob in enumerate(bloblist):
            blob = blob.lower()
            dta = ''
            for x in range(0, len(blob.words)):
                dta += (blob.words[x] + ' ')
            bloblist2.append(dta)

        for i, blob in enumerate(bloblist):
            blob = blob.lower()
            myBlob = []

            for x in range(0, len(blob.words) - 1):
                myBlob.append(blob.words[x] + " " + blob.words[x + 1])

            scores = {word: self.__tfidf_2(word, myBlob, bloblist2) for word in myBlob}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            self.binary_List.append(sorted_words)

    def __triple(self, bloblist):

        for i, blob in enumerate(bloblist):
            blob = blob.lower()
            myBlob = []
            for x in range(0, len(blob.words) - 2):
                myBlob.append(blob.words[x] + " " + blob.words[x + 1] + " " + blob.words[x + 2])

            scores = {word: self.__tfidf_2(word, myBlob, bloblist) for word in myBlob}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            self.three_List.append(sorted_words)

    def __keyword_show(self):

        for i in range(0, len(self.url_list)):
            print self.url_list[i] + " 1-NGram"
            for word, score in self.single_List[i]:
                print word, round(score, 5)

            print self.url_list[i] + " 2-NGram"
            for word, score in self.binary_List[i]:
                print word, round(score, 5)

            print self.url_list[i] + " 3-NGram"
            for word, score in self.three_List[i]:
                print word, round(score, 5)

    def punctuation_clean(self):

        bloblist = []
        for words in self.wordlist:
            bloblist.append(tb(words))
        return bloblist
