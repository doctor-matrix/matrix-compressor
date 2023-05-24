#!python
import os
import pathlib
import sys

import numpy as np
import openai
from dynaconf import Dynaconf
from ratelimit import RateLimitException, limits, sleep_and_retry
from tqdm import tqdm

ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 60


@sleep_and_retry
@limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def main():
    input_filename = pathlib.Path(sys.argv[1])
    output_filename = input_filename.parent / f"{input_filename.stem}.embeddings"

    with open(input_filename) as f:
        data = f.read().splitlines()

    output_array = []
    for line in tqdm(data):
        embedding = get_embedding(line)
        output_array.append(embedding)
    savepoint = np.array(output_array)
    np.savez_compressed(output_filename, embeddings=savepoint)


if __name__ == "__main__":
    main()
