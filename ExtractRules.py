import os
import sys
import numpy as np
from Lib.Tree import SQLiteTree, SQLiteStack, RegularStack, RegularTree
from pathlib import Path
from datetime import datetime
import pickle
import configparser
import threading
import re
import logging
from timeit import default_timer as timer

'''
Please note that the code below uses:
    <d> to denote the dash that separates morphemes
    <q> to denote the question that points to alignment gaps
'''

USE_THREADS = False
PROCESS_LANGS_CONCURRENT = False
PRINT_LOG = False
lock = threading.Lock()

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
    except Exception as err:
        print('get_words_and_morphemes() failed for file', fname)
    return word_and_morphemes


def get_score(top, bottom):
    score = 0
    for i in range(max(len(top), len(bottom))):
        if i < len(top) and i < len(bottom):
            if top[i] == bottom[i]:
                score += 1
    return score


def complete_tree(curr_tree_stack, tree):
    while not curr_tree_stack.is_empty():
        head_of_stack = curr_tree_stack.pop()
        top = head_of_stack[0]
        bottom = head_of_stack[1]
        parent_id = head_of_stack[2]

        if len(top) != 0 and len(bottom) != 0:
            wi = top[0]
            mi = bottom[0]
            if wi == mi:
                top = top[1:]
                bottom = bottom[1:]
                node_id = tree.get_node_id()

                node_t = tree.get_top_of_node(parent_id) + [wi]
                node_b = tree.get_bottom_of_node(parent_id) + [mi]
                node_depth = tree.get_depth_of_node(parent_id) + 1
                tree.add_node(node_t, node_b, node_depth, parent_id, node_id)

                curr_tree_stack.push([top, bottom, node_id])
            elif wi != mi:
                t0 = tree.get_top_of_node(parent_id) + [wi]
                b0 = tree.get_bottom_of_node(parent_id) + ['<q>']
                t1 = tree.get_top_of_node(parent_id) + ['<q>']
                b1 = tree.get_bottom_of_node(parent_id) + [mi]

                score0 = get_score(t0, b0)
                score1 = get_score(t1, b1)

                tc_zero = top[1:]
                bc_zero = bottom
                tc_one = top
                bc_one = bottom[1:]

                node_depth = tree.get_depth_of_node(parent_id) + 1

                if score0 > score1:
                    node_zero_id = tree.get_node_id()
                    tree.add_node(t0, b0, node_depth, parent_id, node_zero_id)
                    curr_tree_stack.push([tc_zero, bc_zero, node_zero_id])
                elif score0 < score1:
                    node_one_id = tree.get_node_id()
                    tree.add_node(t1, b1, node_depth, parent_id, node_one_id)
                    curr_tree_stack.push([tc_one, bc_one, node_one_id])
                else:
                    node_zero_id = tree.get_node_id()
                    node_one_id = tree.get_node_id()

                    tree.add_node(t0, b0, node_depth, parent_id, node_zero_id)
                    tree.add_node(t1, b1, node_depth, parent_id, node_one_id)

                    curr_tree_stack.push([tc_zero, bc_zero, node_zero_id])
                    curr_tree_stack.push([tc_one, bc_one, node_one_id])


def get_max_nodes(all_nodes):
    max_nodes_container = []
    for node in all_nodes:
        node_t = node.get_top()
        node_b = node.get_bottom()
        node_score = get_score(node_t, node_b)

        if len(max_nodes_container) > 0:
            max_node_t = max_nodes_container[0].get_top()
            max_node_b = max_nodes_container[0].get_bottom()
            max_score = get_score(max_node_t, max_node_b)
            if max_score < node_score:
                max_nodes_container.clear()
                max_nodes_container.append(node)
            elif max_score == node_score:
                max_nodes_container.append(node)
        else:
            max_nodes_container.append(node)
    return max_nodes_container


def get_candidate_alignments(w, m):
    path_tree = RegularTree()
    root = path_tree.get_root()

    logging.info('\t\t\t\t\tConstructing alignment tree')
    curr_tree_stack = RegularStack()
    curr_tree_stack.push([w.tolist(), m.tolist(), root.id])
    start_ct_time = timer()
    complete_tree(curr_tree_stack, path_tree)
    end_ct_time = timer()
    logging.info('\t\t\t\t\tDone!')
    logging.debug(
        'Tree construction took {0} for max w/m length {1}'.format(prettify_time(int(end_ct_time - start_ct_time)),
                                                                   max(w.shape[0], m.shape[0])))
    logging.info('\t\t\t\t\tRetrieving max nodes')

    all_end_nodes = path_tree.get_leafs()
    max_nodes = get_max_nodes(all_end_nodes)
    logging.info('\t\t\t\t\tDone!')
    logging.info('\t\t\t\tRetrieving alignments'.format())

    alignments = []
    for node in max_nodes:
        t = node.get_top()
        b = node.get_bottom()

        alignment = np.array((t, b))
        alignments.append(alignment)

    curr_tree_stack.dispose()
    path_tree.dispose()

    logging.info('\t\t\t\tFound {} alignments'.format(len(alignments)))
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

    lock.acquire()
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
    lock.release()
    logging.info('\t\t\tDone!')
    extract_proc_end_time = timer()
    logging.debug(
        'Rule extraction took {0} for max w/m length {1}'.format(
            prettify_time(int(extract_proc_end_time - extract_proc_start_time)),
            max(w.shape[0], m.shape[0])))


def extract_phon_cond_from_file(filename_count, fnames, fname, raw_path, rules_path):
    logging.info('\tProcessing file {0}/{1} - {2}\n'.format(filename_count, len(fnames), fname))
    outpfname = fname.replace(raw_path, rules_path)

    threads = []

    items = get_words_and_morphemes(fname)
    item_count = 0
    for item in items:
        item_count += 1

        w, m = item
        if w.shape[0] <= 10:
            logging.info('\t\t\t{0}. Processing w = {1} and m = {2}'.format(item_count, w, m))
            if USE_THREADS:
                t = threading.Thread(target=extract_phon_cond_from_word_morphemes, args=(w, m, outpfname,))
                t.start()
                threads.append(t)
            else:
                extract_phon_cond_from_word_morphemes(w, m, outpfname)

    if USE_THREADS:
        for t in threads:
            t.join()

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
    USE_THREADS = config.getboolean('DEFAULT', 'THREADING')
    PRINT_LOG = config.getboolean('DEFAULT', 'PRINT_LOG')
    PROCESS_LANGS_CONCURRENT = config.getboolean('DEFAULT', 'PROCESS_LANGS_CONCURRENT')

    if USE_THREADS and PROCESS_LANGS_CONCURRENT:
        threads = []

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
        if USE_THREADS and PROCESS_LANGS_CONCURRENT:
            t = threading.Thread(target=process_lang, args=(lang_count, langs, language,))
            t.start()
            threads.append(t)
        else:
            process_lang(lang_count, langs, language)
        lang_count += 1

    if USE_THREADS and PROCESS_LANGS_CONCURRENT:
        for t in threads:
            t.join()
    logging.info('Ended at {0}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
