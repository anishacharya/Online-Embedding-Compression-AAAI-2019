#!/usr/bin/env python
# -*- coding: utf-8 -*-
mapping = {
    'okay':'neutral',
    'blah':'neutral',
    'blank':'neutral',

    'accomplished': 'anticipation',
    'busy': 'anticipation',
    'creative': 'anticipation',
    'awake': 'anticipation',

    'aggravated': 'anger',
    'annoyed': 'anger',
    'frustrated': 'anger',
    'pissed_off': 'anger',

    'anxious': 'fear',

    'happy':'joy',
    'amused':'joy',
    'cheerful':'joy',
    'chipper':'joy',
    'ecstatic':'joy',
    'excited':'joy',
    'good':'joy',
    'loved':'joy',
    'hopeful':'joy',
    'calm':'joy',
    'content':'joy',
    'crazy':'joy',
    'bouncy':'joy',

    'sad': 'sadness',
    'bored': 'sadness',
    'crappy': 'sadness',
    'crushed': 'sadness',
    'depressed': 'sadness',
    'lonely': 'sadness',
    'contemplative': 'sadness',
    'confused': 'sadness',

    'cold':'tired',
    'exhausted':'tired',
    'drained':'tired',
    'tired':'tired',
    'sleepy':'tired',
    'hungry':'tired',
    'sick':'tired',
}

def doit(dataframe):

    index = []
    statistics = {emotion: 0 for emotion in set(mapping.values())}

    # for i, row in enumerate(dataframe.iterrows()):
    for idx, row in dataframe.iterrows():
        s = row['sentence']
        e = row['label']
        wordcount = len(s.split())
        if wordcount>150:
            continue
        elif e in mapping.keys():
            index.append(idx)
            row['label'] = mapping[e]
            statistics[mapping[e]]+=1

    print statistics
    return dataframe.iloc[index]
