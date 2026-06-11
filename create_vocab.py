import json
from tokenize import build_vocab
stoi, itos = build_vocab('shakespeare_works.txt')
json.dump({'stoi': stoi, 'itos': itos}, open('shakespeare_works_vocab.json', 'w'))
print('Vocab:', len(stoi) - 2, 'words')
