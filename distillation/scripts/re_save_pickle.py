import pickle5 as pickle
import argparse
import logging

import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--data_file", type=str, default="data/dump.txt",
                        help="The path to the data.")
    parser.add_argument("--dump_file", type=str, default="data/",
                        help="The dump file prefix.")
    args = parser.parse_args()
    data = []
    with open(args.data_file, "rb") as fp:
        count = 0
        while True:
            try:
                count += 1
                logger.info(f"starting with {count}")
                new = pickle.load(fp)
                data.extend(new)
                logger.info(f"finished")
            except EOFError:
                break
    dp_file = f"{args.dump_file}.all_pickle.pickle"
    with open(dp_file, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("end")


if __name__ == "__main__":
    main()
