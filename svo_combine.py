import spacy
from svo_extraction import findSVOs
from tqdm import tqdm
import sys

svo_files = ["lm_train_svo_1514261.txt", "lm_train_svo_3028523.txt", "lm_train_svo_4542784.txt", "lm_train_svo_6057047.txt"]
output_file = "lm_train_svo_pre.txt"
with open(output_file,"w") as wp:
    for svo_f in svo_files:
        with open(svo_f,"r") as fp:
            for line in fp:
                wp.write(line)
