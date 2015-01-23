import string
import re
import math
import nltk


class Resources(object):

    @staticmethod
    def set_config(conf):
        Resources.conf = conf

    """ Thresholds """
    adverb_threshold = math.log(500000)

    punctuation = set(string.punctuation)
    punct_re = re.compile("\W+", re.UNICODE)
    num_re = re.compile(r'^([0-9][0-9.,]*)([mMkK]?)$', re.UNICODE)
    pronouns = {
        'me': 'i', 'my': 'i',
        'your': 'you',
        'him': 'he', 'his': 'he',
        'her': 'she',
        'us': 'we', 'our': 'we',
        'them': 'they', 'their': 'they',
    }
    written_numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    stopwords = set(nltk.corpus.stopwords.words('english')) - set(pronouns.iterkeys())
    _global_freqs = None
    _adverb_cache = {}

    @staticmethod
    def is_pronoun_equivalent(word1, word2):
        l1 = word1.lower()
        l2 = word2.lower()
        if Resources.pronouns.get(l1, '') == l2 or Resources.pronouns.get(l2, '') == l1:
            return True
        return False

    @staticmethod
    def get_global_freq(lookup):
        if not Resources._global_freqs:
            Resources._global_freqs = {}
            with open(Resources.conf.get('global', 'freqs')) as f:
                for l in f:
                    try:
                        fd = l.decode('utf8').strip().split(' ')
                        word = fd[1]
                        logfreq = math.log(int(fd[0]) + 2)
                        Resources._global_freqs[word] = logfreq
                    except (ValueError, IndexError):
                        continue
        return Resources._global_freqs.get(lookup, 2)

    @staticmethod
    def is_frequent_adverb(word, pos):
        if not word in Resources._adverb_cache:
            ans = (pos is not None and pos[:2] == 'RB' and Resources.get_global_freq(word) > Resources.adverb_threshold)
            Resources._adverb_cache[word] = ans
        return Resources._adverb_cache[word]

    @staticmethod
    def is_num_equivalent(word1, word2):
        num1 = Resources.to_num(word1)
        num2 = Resources.to_num(word1)
        if num1 and num2:
            return num1 == num2
        return False

    @staticmethod
    def to_num(word):
        if word in Resources.written_numbers:
            return Resources.written_numbers[word]
        m = Resources.num_re.match(word)
        if not m:
            return False
        num = float(m.group(1).replace(',', ''))
        if m.group(2):
            c = m.group(2).lower()
            if c == 'k':
                num *= 1000
            else:
                num *= 1000000
        return num
