from numpy import linalg, array
from itertools import product
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from ast import literal_eval

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
                 test_data, feat_select_thr=0.0, feats={}, kernel='poly',
                 degree=2):
         self.model_name = model_name
         self.train_data = train_data
         self.train_labels = train_labels
         self.test_data = test_data
         self.feat_select_thr = feat_select_thr
         self.feats = feats
         self.kernel = kernel
         self.degree = degree
         self.selector=None
         self.selected_feats = None

    def get_selected_feats(self, support):
        self.selected_feats = {}
        if self.feats != {}:
            reversed_feats = dict([(v,k) for k,v in self.feats.iteritems()])
            for new, old in enumerate(support):
                feat = reversed_feats[old]
                self.selected_feats[feat] = new

    def preproc_train(self):
        if self.feat_select_thr != None:
            self.selector = VarianceThreshold(
                threshold=self.feat_select_thr)
            self.selector = self.selector.fit(
                self.preprocessed_data, self.train_labels)
            self.preprocessed_data = self.selector.transform(self.train_data) 
            self.get_selected_feats(self.selector.get_support(indices=True))

    def preproc_and_train(self):
        self.preprocessed_data = self.train_data
        self.preproc_train()
        self.train(self.preprocessed_data)
    
    def train(self, data):
        if self.model_name == 'linalg_lstsq':
            self.model = linalg.lstsq(data, self.train_labels)[0]
        else:    
            if self.model_name == 'sklearn_linear':
                self.model = linear_model.LinearRegression()
            if self.model_name == 'sklearn_ridge':
                self.model = linear_model.Ridge()
            if self.model_name == 'sklearn_lasso':
                self.model = linear_model.Lasso(alpha=0.001)
            if self.model_name == 'sklearn_elastic_net':
                self.model = linear_model.ElasticNet(alpha=0.001)
            if self.model_name == 'sklearn_kernel_ridge':
                self.model = kernel_ridge.KernelRidge(
                alpha=2, kernel=self.kernel, gamma=None,
                degree=int(self.degree), coef0=1, kernel_params=None)
            if self.model_name == 'sklearn_svr':
                self.model = svm.SVR(kernel=self.kernel,
                                     degree=int(self.degree), coef0=1)
            self.model.fit(data, self.train_labels)
    
    def preproc_test(self, data):
        if self.feat_select_thr != None:
            return self.selector.transform(data)
        else:
            return data

    def preproc_and_predict(self, data):
        preprocessed_data = self.preproc_test(data)
        return self.predict(preprocessed_data)

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
         for item in conf.items('ml') + conf.items('regression'):
             self.set_attribute(item[0], item[1])
         if self.experiment == 'true':
             self.experiment_options = [literal_eval(i[1]) for i
                                        in conf.items('experiment')]
         
    def set_attribute(self, name, value):
        if name == 'train':
            self.train_fn = value
        if name == 'train_labels':
            self.train_labels_fn = value
        if name == 'test':
            self.test_fn = value
        if name == 'gold':
            self.gold_labels_fn = value
        if name == 'model_name':
            self.model_name = value
        if name == 'load_model':
            self.load_model = value
        if name == 'load_model_fn':
            self.load_model_fn = value 
        if name == 'dump_model':
            self.dump_model = value
        if name == 'dump_model_fn':
            self.dump_model_fn = value
        if name == 'feat_select':
            self.feat_select = value
        if name == 'feat_select_thr':
            self.feat_select_thr = value
        if name == 'binary_labels':
            self.binary_labels = value
        if name == 'outfile':
            self.outfile_fn = value
        if name == 'kernel':
            self.kernel = value
        if name == 'degree':
            self.degree = value
        if name == 'experiment': 
            self.experiment = value


    def set_exp_params(self, list_):
        for d in list_:
            for k in d:
                v = d[k]
                self.set_attribute(k, v)

    def regression(self):
        # feature or load model and use its data
        self.get_training_setup()
        
        if self.experiment == 'true':
            for exp_param in product(*self.experiment_options):
                logging.info('experimenting with option {}'.format(repr(exp_param)))
                self.set_exp_params(exp_param)
                self.regression_item(dump_predicted_labels=False)
        else:
            # pass training parameters, train and evaluate the model
            self.regression_item()
        
        if self.dump_model == 'true':
            logging.info('dumping featurized data...')
            with open(self.dump_model_fn, 'w') as f:
                cPickle.dump(self.regression_model, f)
            
    def regression_item(self, dump_predicted_labels=True):
        self.pass_regression_params()
        logging.info('training model...')
        self.regression_model.preproc_and_train()
        logging.info('predicting...')
        predicted = self.regression_model.preproc_and_predict(
            self.regression_model.test_data)
        with open(self.gold_labels_fn) as f:
                self.gold_labels = self.read_labels(f)
        logging.info('correlation on test data:{0}'.format(
            repr(pearsonr(predicted, self.gold_labels))))
        if dump_predicted_labels:
            with open(self.outfile_fn, 'w') as f:
                f.write('\n'.join(str(i) for i in predicted) + '\n')

    def get_training_setup(self):

        if self.load_model == 'true':
            logging.info('loading featurized data...')
            self.regression_model = cPickle.load(open(self.load_model_fn))

        else:
            reader = ReadAndEnrich(self.conf)
            aligner = AlignAndPenalize(self.conf)
            self.featurizer = Featurizer(self.conf, reader, aligner)

            logging.info('featurizing train...')
            with open(self.train_fn) as f:
                train = self.featurizer.featurize(f)
            with open(self.train_labels_fn) as f:
                self.train_labels = self.read_labels(f)
            self.featurizer.reader.clear_pairs()
            logging.info('featurizing test...')
            with open(self.test_fn) as f:
                test = self.featurizer.featurize(f)
            logging.info('converting...')    
            train_feats = self.convert_to_table(train)
            test_feats = self.convert_to_table(test)
            self.regression_model = RegressionModel(
                self.model_name, train_feats, self.train_labels, test_feats)
    
    def pass_regression_params(self):
            self.regression_model.model_name = self.model_name
            self.regression_model.conf = self.conf
            if self.load_model_fn == 'false':
                self.regression_model.feats = self._feat_order
            if hasattr(self, 'kernel'):
                self.regression_model.kernel = self.kernel
            if hasattr(self, 'degree'):
                self.regression_model.degree = self.degree
            if self.feat_select == 'true':
                self.regression_model.feat_select_thr =\
                    float(self.feat_select_thr)
            else:
                self.regression_model.feat_select_thr = None

    def read_labels(self, stream, true_th=0.5):
        labels = []
        for l in stream:
            f = float(l.strip().split('\t')[-1])
            if self.binary_labels == 'true':
                f = 0 if f < true_th else 1
            labels.append(f)
        return array(labels)

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
