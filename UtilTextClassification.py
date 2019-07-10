import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import math
import seaborn as sns



def bin_cut(df, col, bin_range):
    """Specify customized bin range for discretization.

    RETURN:
        target_bin: dataframe of original target value and binned target value.

    """
    target_bin = pd.cut(df[col].astype('float'),
                        bin_range, duplicates='drop')  # drop off bins with the same index, incur less bin number

    print('Share of Each Bin:')
    bin_share = target_bin.groupby(target_bin).agg({'size': lambda x: x.size,
                                                    'share': lambda x: x.size / len(target_bin)})
    display(bin_share)

    map_class = {}
    for i, key in enumerate(sorted(target_bin.unique())):
        map_class[key] = i
    print('Bin and Class Label Correspondence:')
    display(map_class)

    target_bin_2 = target_bin.replace(map_class)
    target_bin = pd.concat([target_bin, target_bin_2], axis=1)
    target_bin.columns = ['{}_bin'.format(col), '{}_class'.format(col)]

    print('\nPreview of return dataframe:')
    display(target_bin.head())

    return target_bin


def concat_str_col(df, *cols, return_col_name='Text', sep_punc=' /// '):
    """
    Concatenate multiple string columns,
    and return df with the concatenated column.

    :param df: dataframe
    :param cols: names of columns
    :param return_col_name: returned concatenated column name
    :param sep_punc: separate punctuation for each column
    :return:
        df: dataframe including concatenated column
    """

    column = cols[0]
    df[return_col_name] = df[column].apply(lambda x: x.strip())

    for column in cols[1:]:
        df[return_col_name] = df[[return_col_name, column]].apply(
            lambda col: sep_punc.join(col.astype(str).str.strip()), axis=1)

    return df


# Plot frequency.
def plot_freq(df, col, top_classes=20):
    """
    :param df: dataframe
    :param col: list of label string
    :param top_classes: (integer) Plot top labels only.
    """
    sns.set_style('whitegrid')

    col = col
    data = df[~df[col].isnull().any(axis=1)]
    data = data.set_index(col)
    
    # Check out the frequency over each concept.
    freq = pd.DataFrame({
            'freq': data.index.value_counts(normalize=True),
            'count': data.index.value_counts(normalize=False)},
            index=data.index.value_counts(normalize=True).index)
    print('Frequency(Top {})...'.format(top_classes))
    freq = freq[:top_classes]
    display(freq)
    
    # Plot bar chart.
    fig, ax = plt.subplots(1,1, figsize=(15,8))
    _ = freq.plot(y='freq', kind='bar', ax=ax, legend=False, colormap='Set2')
    _ = ax.set_ylabel('frequency', fontsize='x-large')
    _ = ax.set_xticklabels(freq.index.values, rotation=40, ha='right')
    _ = ax.set_title('Frequency over Each Class', fontsize='x-large')


# Create sampling dataset.
def split(df, col, col_val, train_num, valid_num, test_num):
    """
    :param col: string
    :param col_val: string
    :return:
        train: dataframe
        valid: dataframe
        test: dataframe
    """
    df = df[df[col] == col_val]
    df = shuffle(df, random_state=1) # shuffle dataset
    train = df.iloc[:train_num, :]
    valid = df.iloc[train_num:train_num + valid_num, :]
    test = df.iloc[train_num + valid_num:train_num + valid_num + test_num, :]
    return train, valid, test


def df2list(text_df, label_df):
    ls_ = [(text_df.iloc[i], {'cats': label_df.iloc[i].to_dict()}) for i in range(len(text_df))]
    return ls_


# Evaluate the model.
def evaluate(nlp, texts, labels, map_ls, label_names):
    """
    :param nlp: spacy nlp object
    :param texts: list of sentences
    :param labels: dictionary of labels
    :param map_ls: dictionary of label to ids
    :param label_names: list of label names
    """
    map_ls = map_ls
    label_names = label_names
    true_labels = []
    pdt_labels = []
    docs = [nlp.tokenizer(text) for text in texts]
    textcat = nlp.get_pipe('textcat')
    for j, doc in enumerate(textcat.pipe(docs)):
        true_series = pd.Series(labels[j]['cats'])
        true_label = true_series.idxmax()  # idxmax() is the new version of argmax()
        # true_label = map_ls[true_series.argmax()]
        true_labels.append(true_label)
        pdt_series = pd.Series(doc.cats)
        pdt_label = pdt_series.idxmax()
        # pdt_label = map_ls[pdt_series.argmax()]
        pdt_labels.append(pdt_label)
    score_f1 = f1_score(true_labels, pdt_labels, average='weighted')
    score_ac = accuracy_score(true_labels, pdt_labels)
    print('f1 score: {:.3f}\taccuracy: {:.3f}'.format(
        score_f1, score_ac))

    print('\nclassification report...')
    print(classification_report(true_labels, pdt_labels, target_names=label_names))


def wide2long(df, map_ls):
    """
    Wide dtaframe to long series.

    :param df: dataframe
    :param map_ls: dictionary of (key,value) mapping
    :return:
        series_: series
    """
    dic_ = df.apply(lambda row: row.to_dict(), axis=1)
    series_ = pd.Series([map_ls[pd.Series(dic_[i]).argmax()] for i in range(len(dic_))])
    return series_


def sk_evaluate(model, feature, label, label_names):
    pred = model.predict(feature)
    true = np.array(label)

    print('Score on dataset...\n')
    print('Confusion Matrix:\n', confusion_matrix(true, pred))
    print('\nClassification Report:\n', classification_report(true, pred, target_names=label_names))
    print('\naccuracy: {:.3f}'.format(accuracy_score(true, pred)))
    print('f1 score: {:.3f}'.format(f1_score(true, pred, average='weighted')))

    return pred, true


def split_size(df, train=0.5, valid=0.3):
    train_size = math.floor(len(df) * train)
    valid_size = math.floor(len(df) * valid)
    test_size = len(df) - train_size - valid_size
    return train_size, valid_size, test_size

