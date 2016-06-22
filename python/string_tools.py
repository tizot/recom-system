import numpy as np
import nltk
import re


class WordHasher(object):
    def __init__(self, n=3, bord='#'):
        self.n_ = n
        self.bord_ = bord
        self.ngrams_ = []

    def init_ngrams(self, tokens):
        ngrams = set()

        for t in tokens:
            # Surround each token with `bord` char
            t = self.bord_ + t + self.bord_

            # Add the ngrams in the token
            for i in range(len(t) - self.n_):
                ngrams.add(t[i:i+self.n_])

        self.ngrams_ = list(sorted(ngrams))

    def load_ngrams(self, ngrams):
        if len(ngrams[0]) == self.n_:
            self.ngrams_ = ngrams
        else:
            raise ValueError("Incompatible ngram sizes (n = %d expected, but got %d)" % (self.n_, len(ngrams[0])))

    def print_ngrams(self):
        print(self.ngrams_)

    def hash(self, s):
        """
            :return: A n-gram count representation of the string given in input.
            :rtype: np.ndarray
        """
        # init counts
        counts = {}
        for g in self.ngrams_:
            counts[g] = 0

        # clean string
        s = StringCleaner().clean_string(s)

        # tokenize string
        pattern = r"(?:[A-Z]\.)+|\w+(?:-\w+)*|\d+(?:\.\d+)?%?"
        sl = set(nltk.regexp_tokenize(s, pattern)) - set(nltk.corpus.stopwords.words("english"))
        num_re = re.compile("^\d+$")
        sl = set([t for t in sl if not num_re.match(t)]) # we remove only-numeric tokens

        # stem tokens
        porter = nltk.stem.PorterStemmer()
        sl = [porter.stem(t) for t in sl]

        # we assume here that the string is clean
        # add bord char around each word
        for a in sl:
            a = self.bord_ + a + self.bord_

        # hash words and increment counts
        for a in sl:
            for i in range(len(a) - self.n_):
                ngram = a[i:i+self.n_]
                if ngram in counts.keys():
                    counts[ngram] += 1

        # convert result into ndarray
        res = np.zeros(len(self.ngrams_))
        for j in range(len(self.ngrams_)):
            res[j] = counts[self.ngrams_[j]]

        return res;


class StringHasher(object):
    def __init__(self, n=1):
        self.n_ = n
        self.ngrams_ = []

    def init_ngrams(self, tokens):
        self.ngrams_ = list(sorted(set(tokens)))

    def load_ngrams(self, ngrams):
        if len(ngrams[0].split(' ')) == self.n_:
            self.ngrams_ = ngrams
        else:
            raise ValueError("Incompatible ngram sizes (n = %d expected, but got %d)" % (self.n_, len(ngrams[0])))

    def print_ngrams(self):
        print(self.ngrams_)

    def hash(self, s):
        """
            :return: A n-gram count representation of the string given in input.
            :rtype: np.ndarray
        """
        # init counts
        counts = {}
        for g in self.ngrams_:
            counts[g] = 0

        # clean string
        s = StringCleaner().clean_string(s)

        # tokenize string
        pattern = r"(?:[A-Z]\.)+|\w+(?:-\w+)*|\d+(?:\.\d+)?%?"
        sl = set(nltk.regexp_tokenize(s, pattern)) - set(nltk.corpus.stopwords.words("english"))
        num_re = re.compile("^\d+$")
        sl = set([t for t in sl if not num_re.match(t)]) # we remove only-numeric tokens

        # stem tokens
        porter = nltk.stem.PorterStemmer()
        sl = [porter.stem(t) for t in sl]

        # we assume here that the string is clean
        # increment counts
        for a in sl:
            if a in counts.keys():
                counts[a] += 1

        # convert result into ndarray
        res = np.zeros(len(self.ngrams_))
        for j in range(len(self.ngrams_)):
            res[j] = counts[self.ngrams_[j]]

        return res;


class StringCleaner(object):
    def __init__(self):
        self.diacriticLetters_ = "àáâãāăȧäảåǎȁȃąạḁẚæǽǣḅḇƀćĉċčƈçḉȼḋɗḍḏḑḓďđðèéêẽēĕėëẻěȅȇẹȩęḙḛǵĝḡğġǧɠģǥĥḣḧȟḥḩḫẖħìíîĩīĭïǐịįȉȋḭɨĳĵǰḱǩḵƙḳķĺḻḷļḽľŀłƚḹµḿṁṃɱɯǹńñṅňŋɲṇņṋṉŉƞòóôõōŏȯöỏőǒȍȏơǫọøǿœṕṗŕṙřȑȓṛŗṟśŝṡšṣșşẗťṭțţṱṯùúûũūŭüủůǔȗụṳųṷṵṽṿẁẃŵẇẅẘẉẋẍỳýŷȳẏÿỷẙźẑżžȥẓẕƶß"
        self.noDiacriticLetters_ = ""
        self.noDiacriticLetters_ += "a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"+"a"
        self.noDiacriticLetters_ += "ae"+"ae"+"ae"
        self.noDiacriticLetters_ += "b"+"b"+"b"
        self.noDiacriticLetters_ += "c"+"c"+"c"+"c"+"c"+"c"+"c"+"c"
        self.noDiacriticLetters_ += "d"+"d"+"d"+"d"+"d"+"d"+"d"+"d"+"d"
        self.noDiacriticLetters_ += "e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"+"e"
        self.noDiacriticLetters_ += "g"+"g"+"g"+"g"+"g"+"g"+"g"+"g"+"g"
        self.noDiacriticLetters_ += "h"+"h"+"h"+"h"+"h"+"h"+"h"+"h"+"h"
        self.noDiacriticLetters_ += "i"+"i"+"i"+"i"+"i"+"i"+"i"+"i"+"i"+"i"+"i"+"i"+"i"+"i"
        self.noDiacriticLetters_ += "ij"
        self.noDiacriticLetters_ += "j"+"j"
        self.noDiacriticLetters_ += "k"+"k"+"k"+"k"+"k"+"k"
        self.noDiacriticLetters_ += "l"+"l"+"l"+"l"+"l"+"l"+"l"+"l"+"l"+"l"
        self.noDiacriticLetters_ += "m"+"m"+"m"+"m"+"m"+"m"
        self.noDiacriticLetters_ += "n"+"n"+"n"+"n"+"n"+"n"+"n"+"n"+"n"+"n"+"n"+"n"+"n"
        self.noDiacriticLetters_ += "o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"+"o"
        self.noDiacriticLetters_ += "oe"
        self.noDiacriticLetters_ += "p"+"p"
        self.noDiacriticLetters_ += "r"+"r"+"r"+"r"+"r"+"r"+"r"+"r"
        self.noDiacriticLetters_ += "s"+"s"+"s"+"s"+"s"+"s"+"s"
        self.noDiacriticLetters_ += "t"+"t"+"t"+"t"+"t"+"t"+"t"
        self.noDiacriticLetters_ += "u"+"u"+"u"+"u"+"u"+"u"+"u"+"u"+"u"+"u"+"u"+"u"+"u"+"u"+"u"+"u"
        self.noDiacriticLetters_ += "v"+"v"
        self.noDiacriticLetters_ += "w"+"w"+"w"+"w"+"w"+"w"+"w"
        self.noDiacriticLetters_ += "x"+"x"
        self.noDiacriticLetters_ += "y"+"y"+"y"+"y"+"y"+"y"+"y"+"y"
        self.noDiacriticLetters_ += "z"+"z"+"z"+"z"+"z"+"z"+"z"+"z"
        self.noDiacriticLetters_ += "s"

    def remove_accents(self, s):
        output = ""
        for c in s:
            try:
                dIndex = self.diacriticLetters_.index(c)
                output += self.noDiacriticLetters_[dIndex]
            except ValueError:
                output += c

        return output

    def clean_string(self, s, bord='#'):
        # lowercase only
        s = str.lower(s)
        # remove accents
        s = self.remove_accents(s)
        # remove bord char
        s = s.replace(bord, '')
        # replace \ by space (LaTeX-like syntax)
        s = s.replace('\\', ' ')

        return s
