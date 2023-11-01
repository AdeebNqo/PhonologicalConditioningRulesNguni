import pickle
import os

def get_wmr_triples(filename):
    triples = []

    f = open(filename, 'rb')
    data = f.read()
    records = data.split(b'[{SEP}]\n')
    if len(records) > 0:
        current_start = 0
        while True:
            w_idx = current_start
            m_idx = current_start + 1
            r_idx = current_start + 2
            current_start = r_idx + 1

            if r_idx >= len(records):
                break
            else:
                w = pickle.loads(records[w_idx])
                m = pickle.loads(records[m_idx])
                r = pickle.loads(records[r_idx])
                triples.append([w, m, r])
    return triples


def get_all_rules(rules_folder, lang_code=None):
    all_rules = {}
    for root, dirs, files in os.walk(rules_folder):
        for direct in dirs:
            triples = []
            adir = os.path.join(root, direct)
            for name in os.listdir(adir):
                filename = os.path.join(adir, name)
                t = get_wmr_triples(filename)
                triples = triples + t
            all_rules[direct] = triples
    return all_rules


if __name__ == '__main__':
    src_folder = 'Rules'
    rs = get_all_rules(src_folder)

    for k, v in rs.items():
        print ('Language = {}'.format(k))
        count = 0
        unq_rules = []
        for trip in v:
            w = trip[0]
            m = trip[1]
            r = trip[2]

            top = r[0,:]
            btm = r[1,:]

            changed_r = '{0}->{1}'.format(''.join([item for item in btm if item != '<q>']), ''.join([item for item in top if item != '<q>']))
            changed_m = ''.join([item if item != '<d>' else '-' for item in m])
            changed_w = ''.join([item for item in w if item != '<q>'])
            if not changed_r in unq_rules:
                unq_rules.append(changed_r)
                print('In combination of {0}, we apply {1}, and get {2}'.format(changed_m, changed_r, changed_w))
            count += 1
        print()