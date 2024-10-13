import re
import pandas as pd
import numpy as np
from stop_words import get_stop_words


# lowercase + base filter
# some basic normalization
def f_base(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r'&gt|&lt', ' ', s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    # normalization 6: string * as delimiter
    s = re.sub(r'\*|\W\*|\*\W', '. ', s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    s = re.sub(r'\(.*?\)', '. ', s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r'\W+?\.', '.', s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 10: ' ing ', noise text
    s = re.sub(r' ing ', ' ', s)
    # normalization 11: noise text
    s = re.sub(r'product received for free[.| ]', ' ', s)
    # normalization 12: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)

    return s.strip()


# filtering out stop words
# create English stop words list

stop_words = (list(
    set(get_stop_words('en'))
    |set(get_stop_words('es'))
    |set(get_stop_words('de'))
    |set(get_stop_words('it'))
    |set(get_stop_words('ca'))
    #|set(get_stop_words('cy'))
    |set(get_stop_words('pt'))
    #|set(get_stop_words('tl'))
    |set(get_stop_words('pl'))
    #|set(get_stop_words('et'))
    |set(get_stop_words('da'))
    |set(get_stop_words('ru'))
    #|set(get_stop_words('so'))
    |set(get_stop_words('sv'))
    |set(get_stop_words('sk'))
    #|set(get_stop_words('cs'))
    |set(get_stop_words('nl'))
    #|set(get_stop_words('sl'))
    #|set(get_stop_words('no'))
    #|set(get_stop_words('zh-cn'))
))

def f_stopw(w_list):
    return [word for word in w_list if word not in stop_words]

def preprocess(texts, samp_size=None):
    """ Preprocess the texts """
    n_docs = len(texts)
    sentences = []
    token_lists = []
    idx_in = []

    if samp_size:
        samp = np.random.choice(n_docs, samp_size)
    else:
        samp = range(n_docs)

    for i, idx in enumerate(samp):
        sentence = f_base(texts[idx])
        token_list = f_stopw(sentence)
        if token_list:
            idx_in.append(idx)
            sentences.append(sentence)
            token_lists.append(token_list)
        print(f'{str(np.round((i + 1) / len(samp) * 100, 2))}')
    print('Preprocessing raw texts. Done!')
    return sentences, token_lists, idx_in

def preprocess_for_siamese(data_df):
    pairs = []
    labels = []
    
    for index, row in data_df.iterrows():
        # Randomly select 50 samples
        sampled_description = data_df['Description'].sample(n=20, random_state=42)
        for index2, row2 in sampled_description.iterrows():
            pairs.append((row['subject_content_text'], row2['Description']))
            if row['Description'] == row2['Description']:
                labels.append(1)
            else:
                labels.append(0)
    pairs_df = pd.DataFrame({'pairs': pairs, 'labels': labels})
    pairs_df = pairs_df.sample(frac=1, random_state=42)