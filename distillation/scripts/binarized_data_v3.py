"""
Preprocessing script before distillation.
"""
import argparse
import logging
import pickle
import random
import time
from os import listdir
import numpy as np

from transformers import BertTokenizer
from multiprocessing import Pool, Value

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)

tokenizer = BertTokenizer.from_pretrained(
    '/data/sedonoso/bert-models/pytorch')
bos = tokenizer.special_tokens_map["cls_token"]  # `[CLS]`
sep = tokenizer.special_tokens_map["sep_token"]  # `[SEP]`


counter = None


def init(arg):
    global counter
    counter = arg


def read_file(file_path):
    all_text =[]
    interval = 10000
    iter = 0
    start = time.time()
    with open(file_path, "r", encoding="utf8") as fp:
        for text in fp:
            all_text.append(text)
            iter += 1
            if iter % interval == 0:
                end = time.time()
                logger.info(
                    f"{iter} examples processed. - {(end - start):.2f}s/{interval}expl")
                start = time.time()
    logger.info(f"readed {iter} lines")
    return all_text


def process_line(text):
    global counter
    text = f"{bos} {text.strip()} {sep}"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = np.uint16(token_ids)
    with counter.get_lock():
        counter.value += 1
    logger.info(f"processed {counter.value}")
    return token_ids


def write_result(results, file_path):
    output_name = file_path.split('.')[0] +'_binarized.pickle'
    with open(output_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Finished pickle")


def main():
    global counter
    ts = time.time()
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--files_path", type=str, default="data/dump.txt",
                        help="The path to the data.")
    args = parser.parse_args()
    logger.info(f"Loading Tokenizer (bert tokenizer)")
    counter = Value('i', 0)
    files_to_process = [args.files_path+t for t in listdir(args.files_path) if not t.startswith('.')]
    for file_path in files_to_process:
        logger.info(f"starting with{file_path} file")
        text_to_process = read_file(file_path)
        time.sleep(3)
        pool = Pool(4, initializer=init, initargs=(counter,))
        results = pool.map(process_line,text_to_process)
        # import ipdb
        # ipdb.set_trace()
        random.shuffle(results)
        write_result(results, file_path)
    logging.info('Took %s', time.time() - ts)


if __name__ == "__main__":
    main()
