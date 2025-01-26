import os
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import configparser
import threading
import re
import logging
from timeit import default_timer as timer
from crpalign import align

'''
Please note that the code below uses:
    <d> to denote the dash that separates morphemes
    <q> to denote the question that points to alignment gaps
'''

PRINT_LOG = False

# Reused. Original source: https://stackoverflow.com/a/2669120/1984350
def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# Reused. Original source: https://stackoverflow.com/a/60144721/1984350
def prettify_time(seconds):
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 3600 % 60
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)

'''
Method for finding the index of the next letter either on the left or right
'''
def get_index_of_letter_in_matrix(start_idx, matrix_2d, move_left=False):
    x_size = matrix_2d.shape[1]
    increment = -1 if move_left else 1
    curr_pos = start_idx
    while -1 < curr_pos and curr_pos <= x_size-1:
        if matrix_2d[1, curr_pos].isalpha():
            return curr_pos
        curr_pos += increment
    return -1

'''
This method retrieves all the file under /data that contain the words and canonical segmentations
'''


def get_filenames(language):
    fnames = []
    for root, dirs, files in os.walk(language):
        for name in files:
            fnames.append(os.path.join(root, name))
    fnames = sorted_nicely(fnames)
    return fnames


'''
This method retrieves all the words and canonical segmentations as numpy arrays.
Specifically, it returns a list of tuples (word, morphemes)
'''


def get_words_and_morphemes(fname):
    word_and_morphemes = []
    f = open(fname, 'r')
    try:
        lines = f.readlines()
        for line in lines:
            items = line.split()
            if len(items) == 2:
                word = np.array(list(items[0]))
                morphs = list(items[1])
                morphs = ['<d>' if x == '/' else x for x in morphs]
                morphemes = np.array(morphs)
                word_and_morphemes.append((word, morphemes))
    except Exception as err7:
        print('get_words_and_morphemes() failed for file', fname)
    return word_and_morphemes


def get_candidate_alignments(w, m):    
    new_m = []
    for _, val in enumerate(m):
        if val == '<d>':
            new_m.append('')
        else:
            new_m.append(val)
            
    lines = [[''.join(w), ''.join(new_m)]]
    _alignments = align.Aligner(lines, '<q>', iterations = 80, mode = 'crp')

    alignments = []
    for iput, oput in _alignments.alignedpairs:       
        t = []
        b = []
        for _, val in enumerate(m):
            if len(iput) > 0 and len(oput) > 0:
                if val == '<d>':
                    t.append('<q>')
                    b.append(val)
                else:
                    if (not '<q>' in [iput[0], oput[0]]) and iput[0] == oput[0]:
                        t.append(iput[0])
                        b.append(oput[0])
                    elif ('<q>' in [iput[0], oput[0]]):
                        t.append(iput[0])
                        b.append(oput[0])
                    else:
                        t.append(iput[0])
                        b.append('<q>')
                        
                        t.append('<q>')
                        b.append(oput[0])
                        
                iput = iput[1:]
                oput = oput[1:]
                  
        alignment = np.array((t, b))
        alignments.append(alignment)
    return alignments


def get_morpheme_spans(alignment):
    spans = []
    current_t_span = []
    current_b_span = []
    for i in range(alignment.shape[1]):
        t = alignment[0, i]
        b = alignment[1, i]
        if b != '<d>':
            current_t_span.append(t)
            current_b_span.append(b)
        else:
            spans.append(np.array([current_t_span, current_b_span]))
            current_t_span = []
            current_b_span = []
    spans.append(np.array([current_t_span, current_b_span]))        
    return spans


def get_phonological_rules(alignments):
    rules = []
    idx = 1
    for alignment in alignments:
        logging.info(
            '\t\t\t\t\tGetting phonological conditioning rules from alignment {0}/{1}'.format(idx, len(alignments)))
        idx += 1
        morpheme_spans = get_morpheme_spans(alignment)
        M = None
        logging.info('\t\t\t\t\tConstructing the M matrix')
        for span in morpheme_spans:
            if not (M is None):
                M = np.concatenate([M, span], axis=1)
            else:
                M = span

        logging.info('\t\t\t\t\tCollecting rules from the matrix {0}'.format(M))
        if not (M is None):
            column_length = M.shape[1]
            start = 0
            while start <= column_length - 1:
                if (M[0, start] == '<q>' and M[1, start].isalpha()) or (M[0, start].isalpha() and M[1, start] == '<q>'):
                    found_end = False
                    for end in range(start + 1, column_length):
                        if M[0, end].isalpha() and M[1, end].isalpha():
                            found_end = True
                            if start != end and end-start > 1:
                                morph_chars = M[1:, start:end][0].tolist()
                                if sum([1 if item.isalpha() else 0 for item in
                                        morph_chars]) > 1:  # If there is more than one letter in the morpheme span
                                    rule = M[:, start:end]
                                    start = end+1
                                    rules.append(rule)
                                    break
                                else:
                                    rule = M[:, start:end+1]
                                    start = end+2
                                    rules.append(rule)
                                    break
                            else:
                                l = get_index_of_letter_in_matrix(start, M, True)
                                r = get_index_of_letter_in_matrix(end, M, False)
                                if l != -1 and r != -1:
                                    rule = M[:, l:r]
                                    if ('<q>' in M[0:, start:end][0].tolist() or '<q>' in M[1:, start:end][0].tolist()):
                                        l = l-1 if l>=1 else l
                                        r = r+1 if r<=M.shape[1]-1 else r
                                        rule = M[:, l:r]
                                        start = r+1
                                        rules.append(rule)
                                        break
                    if not found_end:
                        break
                else:
                    start += 1
        logging.info('\t\t\t\t\tDone!')
    return rules


def extract_phon_cond_from_word_morphemes(w, m, outpfname):
    extract_proc_start_time = timer()
    logging.info('\t\t\t\tAligning w and m')
    alignments = get_candidate_alignments(w, m)
    logging.info('\t\t\t\tRetrieving phonological conditioning rules')
    rules = get_phonological_rules(alignments)
    logging.info('\t\t\t\tFound {0} phonological conditioning rules'.format(len(rules)))
    idx_rule = 0

    p = Path(outpfname)
    if not os.path.exists(outpfname):
        p.parent.mkdir(exist_ok=True, parents=True)
    f = p.open('ab')

    for rule in rules:
        f.write(pickle.dumps(w))
        f.write(b'[{SEP}]\n')
        f.write(pickle.dumps(m))
        f.write(b'[{SEP}]\n')
        f.write(pickle.dumps(rule))
        f.write(b'[{SEP}]\n')
        f.flush()
        logging.info('\t\t\t\t\t{0}/{1}. Rule = {2}'.format(idx_rule + 1, len(rules), rule))
        idx_rule += 1
    f.close()

    logging.info('\t\t\tDone!')
    extract_proc_end_time = timer()
    logging.debug(
        'Rule extraction took {0} for max w/m length {1}'.format(
            prettify_time(int(extract_proc_end_time - extract_proc_start_time)),
            max(w.shape[0], m.shape[0])))


def extract_phon_cond_from_file(filename_count, fnames, fname, raw_path, rules_path):
    logging.info('\tProcessing file {0}/{1} - {2}\n'.format(filename_count, len(fnames), fname))
    outpfname = fname.replace(raw_path, rules_path)

    items = get_words_and_morphemes(fname)
    item_count = 0
    for item in items:
        item_count += 1

        w, m = item
        
        logging.info('\t\t\t{0}. Processing w = {1} and m = {2}'.format(item_count, w, m))
        extract_phon_cond_from_word_morphemes(w, m, outpfname)

    logging.info('Done processing file {0}'.format(fname))


def process_lang(lang_count, langs, language):
    logging.info('Processing language {0}/{1} - {2}\n'.format(lang_count, len(langs), language))

    raw_path = os.path.join('ProcessedData', language)
    rules_path = os.path.join('Rules', language)

    fnames = get_filenames(raw_path)
    filename_count = 1
    for fname in fnames:
        outpfname = fname.replace(raw_path, rules_path)
        if not os.path.exists(outpfname):  # Now trying to have a fake "cache". Do not attempt to recreate rules if
            # they exist
            extract_phon_cond_from_file(filename_count, fnames, fname, raw_path, rules_path)
        filename_count += 1


if __name__ == "__main__":
    # setting config details
    config = configparser.ConfigParser()
    config.read('config.ini')
    PRINT_LOG = config.getboolean('DEFAULT', 'PRINT_LOG')

    logging.basicConfig(filename='output.txt', level=logging.DEBUG)
    if not PRINT_LOG:
        logging.disable(logging.INFO)

    logging.info('Started at {0}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    langs = [
        'nr',
        'ss',
        'xh',
        'zu'
    ]

    lang_count = 1
    for language in langs:
        process_lang(lang_count, langs, language)
        lang_count += 1

    logging.info('Ended at {0}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
