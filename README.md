semeval
=======

Semantic Textual Similarity (STS) system created by the MathLingBudapest team to participate in Tasks 1 and 2 of Semeval2015

_NOTE: this code and its dependency pymachine are under constant development. To run the version of the code that was used to create the MathLingBudapest team's submissions (although many bugs have been fixed since), use the following revisions:_

Task 1: https://github.com/juditacs/semeval/tree/submitted

Task 2: https://github.com/juditacs/semeval/tree/15863ba5bc7f857291322c707a899c7c802a7c88

If you'd also like to reproduce the machine similarity component as it was at the time of the submission, you'll need the following revision of the pymachine repository:

https://github.com/kornai/pymachine/tree/3d936067e775fc8aa56c06388eb328ef2c6efe75


# Dependencies
Use pip to install the numpy and scipy packages. To use the machine similarity component, you need to download and install the pymachine module from https://github.com/kornai/pymachine

__A standard python module that takes care of its dependencies is on its way__

# Usage

The STS system can be invoked from the repo's base directory using:

    cat twitter_test_data | python semeval/paraphrases.py -c configs/twitter.cfg > out

or to use the machine similarity component

    cat twitter_test_data | python semeval/paraphrases.py -c configs/sts_machine.cfg > out


# Regression

## Regression used for Twitter data

Specifying regression mode in the final\_score section uses a regression (see `configs/twimash.cfg`).
This mode needs to know the location of the train and test files, which are specified in the regression section:

    [regression]
    train: data/train.data
    train_labels: data/train.labels
    test: data/test.data
    gold: data/test.label
    binary_labels: true
    outfile: data/predicted.labels

Specifying a gold file is optional, the rest of the options are mandatory.
If you specify a gold file, precision, recall and F-score are computed and printed to stdout.


## Regression used for Task 2 STS data

sample uses of regression.py:

     python scripts/regression.py regression_train all.model semeval_data/sts_trial/201213_all data/1213_all/ngram data/1213_all/lsa data/1213_all/machine

     for f in data/2014-test/nosim_4gr_d/STS.input.*; do topic=`basename $f | cut -d'.' -f3`; echo $topic; python scripts/regression.py regression_predict all.model data/2014-test/nosim_4gr_d/STS.input.$topic.txt.out data/2014-test/lsa_sim_bp/STS.input.$topic.txt.out data/2014-test/machine_sim_nodes2/STS.input.$topic.txt.out > data/2014-test/regr/STS.input.$topic.txt.out; done


For certain scripts to work, you may want to set the environment variables SEMEVAL_DATA and STANFORD_PARSER.
On nessi6, set_env_nessi6.sh does this for you.
