import os
import sys

from liblinear import gen_feature_nodearray, liblinear
from liblinearutil import load_model, parameter, predict, problem, train, save_model  # nopep8
from numpy import linalg


def num_to_class(num):
    return int(10*round(num, 1))

def class_to_num(feat):
    return float(feat) / 10

def get_data(gold_dir, dirs):
    vectors, predicates = [], []
    gold_files = [fn for fn in os.listdir(gold_dir) if ".gs." in fn]
    for fn in gold_files:
        topic = fn.split('.')[2]
        auto_fn = "STS.input.{0}.txt.out".format(topic)
        file_objs = [open(os.path.join(d, auto_fn)) for d in dirs]

        for line in open(os.path.join(gold_dir, fn)):
            predicates.append(float(line.strip()))
            vectors.append([float(f.readline().strip())
                            for f in file_objs])
    return vectors, predicates

def get_gold_from_dir(gold_dir):
    predicates = []
    gold_files = [fn for fn in os.listdir(gold_dir) if ".gs." in fn]
    for fn in gold_files:
        for line in open(os.path.join(gold_dir, fn)):
            predicates.append(float(line.strip()))
    return predicates

def read_feats_from_dir(dirname):
    feats = []
    for fn in os.listdir(dirname):
        feats += read_feats_from_file(os.path.join(dirname, fn))
    return feats

def read_feats_from_file(fn):
    feats = []
    with open(fn) as f:
        for l in f:
            feats.append([float(i) for i in l.strip().split(' ')])
    return feats

def train_regression(args):
    model_name, gold_dir, dirs = args[0], args[1], args[2:]
    #vectors = read_feats_from_dir(feat_dir)
    #predicates = get_gold_from_dir(gold_dir)
    vectors, predicates = get_data(gold_dir, dirs)
    w = linalg.lstsq(vectors, predicates)
    params = w[0]
    with open(model_name, 'w') as outfile:
        outfile.write("\t".join(map(str, params))+"\n")

def train_liblinear(args):
    model_name, gold_dir, dirs = args[0], args[1], args[2:]
    vectors, predicates = get_data(gold_dir, dirs)
    prob = problem(map(num_to_class, predicates), vectors)
    param = parameter('-s 0')
    model = train(prob, param)
    save_model(model_name, model)

def predict_liblinear(args):
    model_name, files = args[0], args[1:]
    model = load_model(model_name)
    file_objs = [open(fn) for fn in files]
    while True:
        lines = [f.readline() for f in file_objs]
        if '' in lines:
            assert len(set(lines)) == 1
            break
        vector = [float(line.strip()) for line in lines]
        xi, max_idx = gen_feature_nodearray(vector)
        label = liblinear.predict(model, xi)
        #p_label, p_acc, p_val = predict([1], [vector], model)
        print class_to_num(label)
        #print class_to_num(p_label[0])

def predict_regression(args):
    model_name, files = args[0], args[1:]
    params = map(float, open(model_name).readline().split("\t"))
    file_objs = [open(fn) for fn in files]
    while True:
        lines = [f.readline() for f in file_objs]
        if '' in lines:
            assert len(set(lines)) == 1
            break
        vector = [float(line.strip()) for line in lines]
        answer = sum(params[i]*x for i, x in enumerate(vector)) / sum(params)
        print answer


def main():
    task = sys.argv[1]
    if task == 'liblinear_train':
        train_liblinear(sys.argv[2:])
    if task == 'regression_train':
        train_regression(sys.argv[2:])
    elif task == 'liblinear_predict':
        predict_liblinear(sys.argv[2:])
    elif task == 'regression_predict':
        predict_regression(sys.argv[2:])
    else:
        raise Exception('unknown task')

if __name__ == '__main__':
    main()
