import logging
import math
import multiprocessing as mp
from collections import ChainMap
from itertools import chain

from log_config import setup_logging
from main import get_model_candidates_from_hf, process_batch, print_report


def main():
    setup_logging()

    logging.info('Started')

    num_cores = mp.cpu_count()
    logging.info(f'Multiprocessing enabled for {num_cores} cores')

    pool = mp.Pool(num_cores)

    models, tags = get_model_candidates_from_hf()

    total_models = len(models)
    chunk_size = num_cores
    ranges = []
    for num in range(math.ceil(total_models / chunk_size)):
        start, end = num * chunk_size, min((num + 1) * chunk_size, total_models)
        ranges.append((start, end))

    ranges = [(0, 10), (11, 22), (23, 100)]
    result_objects = [pool.apply_async(process_batch, args=(models, start, end)) for start, end in ranges]

    results = [r.get() for r in result_objects]

    no_config_models = list(chain.from_iterable(r[0] for r in results))
    missed_type = list(chain.from_iterable(r[2] for r in results))
    not_exportable_models = dict(ChainMap(*[r[1] for r in results]))

    print_report(no_config_models, not_exportable_models, missed_type, models, tags)


if __name__ == '__main__':
    main()
