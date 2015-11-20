from numpy import linalg, array
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
from sentence import SentencePair
import logging
import cPickle
from read_and_enrich import ReadAndEnrich
from align_and_penalize import AlignAndPenalize

class Featurizer(object):

    def __init__(self, conf, reader, aligner):
        self.reader = reader
        self.aligner = aligner
        self.conf = conf

    def featurize(self, stream):
        sample = []
        pairs = self.reader.read_sentences(stream)
        for s1, s2 in pairs:
            pair = SentencePair(s1, s2)
            self.aligner.align(pair)
            sample.append(pair)
        return sample

class RegressionModel:

    def __init__(self, model_name, train_data, train_labels,
                 test_data, feat_select_thr=0.0, feats={}):
         self.model_name = model_name
         self.train_data = train_data
         self.train_labels = train_labels
         self.test_data = test_data
         self.feat_select_thr = feat_select_thr
         self.feats = feats

    
    def get_selected_feats(self):
        self.selected_feats = {}
        if self.feats != {}:
            reversed_feats = dict([(v,k) for k,v in self.feats.iteritems()])
            for new, old in enumerate(self.selector.get_support(indices=True)):
                feat = reversed_feats[old]
                self.selected_feats[feat] = new


    def select_and_train(self):
        if self.feat_select_thr != None:
            self.selector = VarianceThreshold(
                threshold=self.feat_select_thr)
            self.selector = self.selector.fit(
                self.train_data, self.train_labels)
            self.selected_train = self.selector.transform(self.train_data) 
            self.get_selected_feats()
            self.train(self.selected_train)
        else:
            self.selector = None
            self.selected_feats = None
            self.train(self.train_data)

    def train(self, data):
        if self.model_name == 'linalg_lstsq':
            self.model = linalg.lstsq(data, self.train_labels)[0]
        if self.model_name == 'sklearn_linear':
            self.model = linear_model.LinearRegression()
            self.model.fit(data, self.train_labels)
        if self.model_name == 'sklearn_kernel_ridge':
            self.model = kernel_ridge.KernelRidge(
                alpha=2, kernel='polynomial', gamma=None,
                degree=3, coef0=1, kernel_params=None)
            self.model.fit(data, self.train_labels)
        if self.model_name == 'sklearn_svr':
            self.model = svm.SVR(kernel='poly', degree=3, coef0=1)
            self.model.fit(data, self.train_labels)

    def select_and_predict(self, data):
        if self.feat_select_thr != None:
            return self.predict(self.selector.transform(data))
        else:
            return self.predict(data)

    def predict(self, data):
        if self.model_name == 'linalg_lstsq':
            return self.predict_regression(data)
        if self.model_name[:7] == 'sklearn':
            return self.model.predict(data)

    def predict_regression(self, feats, true_th=0.5):
        scores = []
        for sample in feats:
            ans = sum(self.model[i] * x for i, x in enumerate(sample))
            scores.append(ans)
        return scores


class Regression(object):

    def __init__(self, conf):
         self.conf = conf
         self._feat_order = {}
         self._feat_i = 0

    def regression(self):
        # featurize /load existing model (with its featurized training and test sets)
        self.get_training_setup()
        logging.info('training model...')
        self.regression_model.select_and_train()
        logging.info('predicting...')
        predicted = self.regression_model.select_and_predict(
            self.regression_model.test_data)
        with open(self.conf.get('regression', 'outfile'), 'w') as f:
            f.write('\n'.join(str(i) for i in predicted) + '\n')
        self.dump_if_needed()

    def get_training_setup(self):

        model_name = self.conf.get('ml', 'model_name')
        if self.conf.get('ml', 'load_model') == 'true':
            logging.info('loading featurized data...')
            self.regression_model = cPickle.load(open(self.conf.get(
                'ml', 'load_model_fn')))
            self.regression_model.model_name = model_name

        else:
            reader = ReadAndEnrich(self.conf)
            aligner = AlignAndPenalize(self.conf)
            self.featurizer = Featurizer(self.conf, reader, aligner)

            logging.info('featurizing train...')
            with open(self.conf.get('regression', 'train')) as f:
                train = self.featurizer.featurize(f)
            with open(self.conf.get('regression', 'train_labels')) as f:
                train_labels = self.read_labels(f)
            self.featurizer.reader.clear_pairs()
            logging.info('featurizing test...')
            with open(self.conf.get('regression', 'test')) as f:
                test = self.featurizer.featurize(f)
            logging.info('converting...')    
            train_feats = self.convert_to_table(train)
            test_feats = self.convert_to_table(test)
            self.regression_model = RegressionModel(
                model_name, train_feats, train_labels, test_feats)
            # model stores config data so that it is possible to reproduce featurizing
            self.regression_model.conf = self.conf
            self.regression_model.feats = self._feat_order
        if self.conf.get('ml', 'feat_select') == 'true':
            self.regression_model.feat_select_thr =\
                    float(self.conf.get('ml', 'feat_select_thr'))
        else:
            self.regression_model.feat_select_thr = None


    def dump_if_needed(self):

        if self.conf.get('ml', 'dump_model') == 'true':
            logging.info('dumping featurized data...')
            with open(self.conf.get('ml', 'dump_model_fn'), 'w') as f:
                cPickle.dump(self.regression_model, f)

    def read_labels(self, stream, true_th=0.5):
        labels = []
        for l in stream:
            f = float(l.strip().split('\t')[-1])
            if self.conf.getboolean('regression', 'binary_labels'):
                f = 0 if f < true_th else 1
            labels.append(f)
        return labels

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
        return array(table)
