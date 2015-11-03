from numpy import linalg
from sentence import SentencePair
from collections import defaultdict
import logging
import cPickle
from read_and_enrich import ReadAndEnrich
from align_and_penalize import AlignAndPenalize

class Featurizer(object):

    def __init__(self, conf, reader, aligner):
        self.reader = reader
        self.aligner = aligner
        self._feat_order = {}
        self._feat_i = 0

    def featurize(self, stream):
        sample = []
        pairs = self.reader.read_sentences(stream)
        for s1, s2 in pairs:
            pair = SentencePair(s1, s2)
            self.aligner.align(pair)
            sample.append(pair)
        return sample

    def convert_to_table(self, sample):
        table = []
        for s in sample:
            table.append([0] * self._feat_i)
            for feat, sc in s.features.iteritems():
                if not self._feat_order:
                    self._feat_order[feat] = 0
                    self._feat_i = 1
                    table[-1] = [sc]
                elif feat not in self._feat_order:
                    self._feat_order[feat] = self._feat_i
                    self._feat_i += 1
                    table[-1].append(sc)
                else:
                    table[-1][self._feat_order[feat]] = sc

class RegressionModel:

    def __init__(self, model_name, train_data, train_labels,
                 test_data):
         self.model_name = model_name
         self.train_data = train_data
         self.train_labels = train_labels
         self.test_data = test_data

    def train(self):
        if self.model_name == 'linalg_lstsq':
            self.model = linalg.lstsq(self.train_data, self.train_labels)[0]
    
    def predict(self, test_data):
        if self.model_name == 'linalg_lstsq':
            self.predict_regression(self.test_data)
     
    def predict_regression(self, feats, true_th=0.5):
        scores = []
        for sample in feats:
            ans = sum(self.model[i] * x for i, x in enumerate(sample))
            if self.conf.getboolean('regression', 'binary_labels'):
                f = 0 if ans < true_th else 1
                scores.append(f)
            else:
                scores.append(ans)
        return scores
        

class Regression(object):

    def __init__(self, conf):
         self.conf = conf

    

    def regression(self):

        self.get_regression_model()
        self.regression_model.train()
        self.regression_model.predict()
        self.dump_if_needed()
        
    def get_regression_model(self):
        model_name = self.conf.get('ml', 'model_name')
        if self.conf.get('ml', 'load_model') == 'true':
            logging.info('loading featurized data...')
            self.regression_model = cPickle.load('ml', 'load_model_fn')
            self.regression_model.model_name = model_name

        else:
            reader = ReadAndEnrich(self.conf)
            aligner = AlignAndPenalize(self.conf)
            self.featurizer = Featurizer(reader, aligner) 
            logging.info('featurizing train...')
            with open(self.conf.get('regression', 'train')) as f:
                train = self.featurize(f)
            with open(self.conf.get('regression', 'train_labels')) as f:
                train_labels = self.read_labels(f)
            self.reader.clear_pairs()
            logging.info('featurizing test...')
            with open(self.conf.get('regression', 'test')) as f:
                test = self.featurize(f)
            train_feats = self.convert_to_table(train)
            test_feats = self.convert_to_table(test)
            self.regression_model = RegressionModel(model_name, train, train_labels, test) 


        logging.info('converting...')
        self.train_feats = self.convert_to_table(self.train)
        f = open('train_feats', 'w')
        cPickle.dump(self.train_feats, f)
        self.test_feats = self.convert_to_table(self.test)
        # this created matrices whose rows are sens and columns are sim types
        logging.info('running regression...')
        self.model = linalg.lstsq(self.train_feats, self.train_labels)[0]
        # logging.info('model: {0}'.format(self.model))
        logging.info('predicting...')
        self.predicted = self.predict_regression(self.test_feats, 0.3)
        with open(self.conf.get('regression', 'outfile'), 'w') as f:
            f.write('\n'.join(str(i) for i in self.predicted) + '\n')
            # f.write('\n'.join(str(int(i if i < 0.5 else 1.0))
            #                   for i in self.predicted) + '\n')

    def read_labels(self, stream, true_th=0.5):
        labels = []
        for l in stream:
            f = float(l.strip().split('\t')[-1])
            if self.conf.getboolean('regression', 'binary_labels'):
                f = 0 if f < true_th else 1
            labels.append(f)
        return labels
    
