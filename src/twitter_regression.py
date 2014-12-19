from numpy import linalg
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sys import argv
from collections import defaultdict


def read_gold(fn):
    with open(fn) as f:
        return [float(l) for l in f]


def read_feats(fn):
    feats = []
    with open(fn) as f:
        for l in f:
            feats.append([float(i) for i in l.strip().split(' ')])
    return feats


def train_regression(feats, gold):
    w = linalg.lstsq(feats, gold)
    return w[0]


def train_svm(feats, gold):
    labeled = []
    clf = svm.SVC()
    clf.fit(feats, gold)
    return clf
    for i, f in enumerate(feats):
        feat_dict = {}
        for j, feat in enumerate(f):
            feat_dict[j] = feat
        labeled.append([feat_dict, gold[i]])
    return SklearnClassifier(Pipeline()).train(labeled, max_iter=30)


def predict_svm(model, feats):
    return model.predict(feats)
    pred = []
    for f in feats:
        feat_d = {}
        for i, feat in enumerate(f):
            feat_d[i] = feat
        pred.append(model.classify(feat_d))
    return pred


def predict_regression(model, feats):
    scores = []
    for sample in feats:
        ans = sum(model[i] * x for i, x in enumerate(sample))
        scores.append(ans)
    return scores


def print_stats(prediction, gold):
    stat = defaultdict(int)
    true_th = 0.3
    for i, p in enumerate(prediction):
        if p <= true_th:
            if gold[i] < 0.5:
                stat['tn'] += 1
            else:
                stat['fn'] += 1
        else:
            if gold[i] >= 0.5:
                stat['tp'] += 1
            else:
                stat['fp'] += 1
    N = sum(stat.values())
    print('\ntrue positive: {0}\ntrue negative: {1}\nfalse positive: {2}\nfalse negative: {3}\nsum: {4}\n******'.format(stat['tp'], stat['tn'], stat['fp'], stat['fn'], N))
    prec = float(stat['tp']) / (stat['tp'] + stat['fp'])
    rec = float(stat['tp']) / (stat['tp'] + stat['fn'])
    acc = float(stat['tp'] + stat['tn']) / N
    F = 2 * prec * rec / (prec + rec)
    print('Precision: {0}\nRecall: {1}\nF1: {2}\nAccuracy: {3}'.format(prec, rec, F, acc))


def main():
    train_feats = read_feats(argv[1])
    dev_feats = read_feats(argv[2])
    if len(argv) > 3:
        fn = argv[3]
    else:
        fn = 'data/filt/labels_train_binary'
    train_gold = read_gold(fn)
    if len(argv) > 4:
        fn = argv[4]
    else:
        fn = 'data/filt/labels_dev_binary'
    dev_gold = read_gold(fn)
    #model = train_svm(train_feats, train_gold)
    model = train_regression(train_feats, train_gold)
    #prediction = predict_svm(model, dev_feats)
    prediction = predict_regression(model, dev_feats)
    th = 0.25
    with open('predicted', 'w') as f:
        for p in prediction:
            if p >= th:
                if p > 1:
                    p = 1
                f.write('true\t{0:1.4f}\n'.format(round(p, 4)))
            else:
                if p < 0:
                    p = 0
                f.write('false\t{0:1.4f}\n'.format(round(p, 4)))
        f.write('\n'.join(map(str, prediction)) + '\n')
    if dev_gold:
        print_stats(prediction, dev_gold)

if __name__ == '__main__':
    main()
