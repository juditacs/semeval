semeval
=======

MathLing Budapest Team's repo

For scripts to work, set the environment variables SEMEVAL_DATA and STANFORD_PARSER.
On nessi6, set_env_nessi6.sh does this for you.


sample uses of regression.py:

python scripts/regression.py regression_train all.model semeval_data/sts_trial/201213_all data/1213_all/ngram data/1213_all/lsa data/1213_all/machine

for f in data/2014-test/nosim_4gr_d/STS.input.*; do topic=`basename $f | cut -d'.' -f3`; echo $topic; python scripts/regression.py regression_predict all.model data/2014-test/nosim_4gr_d/STS.input.$topic.txt.out data/2014-test/lsa_sim_bp/STS.input.$topic.txt.out data/2014-test/machine_sim_nodes2/STS.input.$topic.txt.out > data/2014-test/regr/STS.input.$topic.txt.out; done


## Twitter regression

     python src/twitter_regression.py data/filt/train_feat data/filt/dev_feat data/filt/labels_train data/filt/labels_dev


This script runs a regression using the train features and labels specified in argv[1] and argv[3] and test them on the dev.
I also implemented an SVM classifier, you need to replace the appropriate lines in the main function.

# New version

The new refactored version can be invoked from the repo's base directory using:

    cat twitter_test_data | python semeval/paraphrases.py -c configs/twitter.cfg > out

Right now it does not produce the submitted results because crucial features are missing.
Once they are implemented, I will try to reproduce the twitter results (without machines).
