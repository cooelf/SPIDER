import spacy
from svo_extraction import findSVOs
from tqdm import tqdm
import sys

nlp = spacy.load("en_core_web_sm")

def pre_lm_finetune(start_id, end_id):
    train_path = "data/ubuntu_data/lm_train.txt"
    svo_train_path = "data/ubuntu_data/lm_train_svo_" + str(end_id) +".txt"
    all_data = []
    print("reading data...")
    with open(train_path, 'r', encoding='utf-8') as rf:
        for line in tqdm(rf):
            all_data.append(line)
    print("processing data...")
    print(svo_train_path)
    print(start_id, end_id, len(all_data), end_id - start_id)
    all_data = all_data[start_id:end_id]
    with open(svo_train_path,'w',encoding='utf-8') as wf:
        for line in tqdm(all_data):
            line = line.strip()
            if line:
                tok = nlp(line)
                svos = findSVOs(tok)
                wf.write(str(svos)+'\n')
            else:
                wf.write('\n')
start_id = int(sys.argv[1])
end_id = int(sys.argv[2])
pre_lm_finetune(start_id, end_id)