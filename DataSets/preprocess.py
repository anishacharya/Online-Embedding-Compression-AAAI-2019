#!/usr/bin/env python

"""Text Classification Preprocessing
"""

import re
import os
import sys
import codecs
import pandas
import argparse
import yaml
import importlib

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"\. \. \.", "\.", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def parse_arg(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', help='dataset name')
    parser.add_argument('-m', '--mapping', default='corpus.yaml', help='mapping yaml file to map corpus name to file')
    return parser.parse_args(argv[1:])


def append_document(dataframe_list, dirname, fnames):
    if dirname=='.':
        return

    sentences = []
    num_files = 0
    for fname in fnames:

        f_path = os.path.join(dirname,fname)
        if os.path.isfile(f_path):
            num_files += 1
            with open(f_path) as f:
                sentences.append(clean_str(f.read()))

    labels = [os.path.basename(dirname)]*num_files 
    dataframe_list.append(pandas.DataFrame({'sentence':sentences, 'label':labels, 'split':'train'}))


if __name__ == '__main__':
    args = parse_arg(sys.argv)
    dataframe_list = []
    with open(args.mapping) as f:
        corpus = yaml.load(f)
        assert 'dir' in corpus, "yaml file should contain 'dir: path/to/data' line"

    dataset = corpus[args.dataset]
    corpus_dir = corpus['dir']

    if 'dir' in dataset:
        corpus_dir = os.path.join(corpus_dir, dataset['dir'])
        for emotion in os.listdir(corpus_dir):
            emotion_dir = os.path.join(corpus_dir, emotion)
            if os.path.isdir(emotion_dir):
                os.path.walk(emotion_dir, append_document, dataframe_list)
    else:
        for split, filename in dataset.items():
            filename = corpus_dir+'/'+filename
            if not filename:
                continue
            labels = []
            sentences = []
            with open(filename) as f:
                for line in f:
                    div = line.index(' ')
                    sentences.append(clean_str(line[div+1:]))
                    labels.append(line[:div])
            dataframe_list.append(pandas.DataFrame({'sentence':sentences, 'label':labels, 'split':split}))

    dataframe = pandas.concat(dataframe_list, ignore_index=True)

    if 'postprocess' in dataset:
        assert 'postprocess_dir' in corpus, 'Please specify postprocessing scripts directories.'
        module = importlib.import_module(corpus['postprocess_dir']+'.'+dataset['postprocess'])
        assert 'doit' in module.__dict__, 'Please implement the doit function (dataframe->dataframe) int {}'.format(dataset['postprocess'])
        dataframe = module.doit(dataframe)

    filename = args.dataset + '.pkl'
    dataframe.to_pickle(filename)
    # pandas.concat(dataframe_list).to_csv(filename+'.csv')
