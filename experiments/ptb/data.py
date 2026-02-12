"""PTB dataset loading with stateful batching for language modeling.

Uses Mikolov-preprocessed Penn Treebank with ~10k vocabulary.
Dataset source: GitHub mirror of Mikolov's RNNLM data.
"""
import collections
import os
import time
import urllib.error
import urllib.request

import numpy as np
import tensorflow as tf


# GitHub mirror of Mikolov's PTB data (reliable)
PTB_GITHUB_BASE = "https://raw.githubusercontent.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/master/data"


def download_ptb(max_retries: int = 3, retry_delay: float = 1.0) -> str:
    """Download PTB dataset, return path to data directory.

    Downloads from GitHub mirror with retry logic for network failures.

    Args:
        max_retries: Maximum download attempts per file
        retry_delay: Seconds between retry attempts

    Returns:
        Path to directory containing ptb.train.txt, ptb.valid.txt, ptb.test.txt

    Raises:
        RuntimeError: If download fails after all retries
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".keras", "datasets", "ptb")
    os.makedirs(cache_dir, exist_ok=True)

    for filename in ["ptb.train.txt", "ptb.valid.txt", "ptb.test.txt"]:
        filepath = os.path.join(cache_dir, filename)
        if not os.path.exists(filepath):
            url = f"{PTB_GITHUB_BASE}/{filename}"
            
            for attempt in range(max_retries):
                try:
                    print(f"Downloading {url}...")
                    urllib.request.urlretrieve(url, filepath)
                    break
                except (urllib.error.URLError, urllib.error.HTTPError) as e:
                    if attempt < max_retries - 1:
                        print(f"Download failed ({e}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        # Clean up partial download
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {e}")

    return cache_dir


def read_words(filename: str) -> list[str]:
    """Read PTB file, splitting into words with <eos> for newlines.

    Args:
        filename: Path to PTB text file

    Returns:
        List of words including <eos> tokens
    """
    with open(filename, encoding="utf-8") as f:
        return f.read().replace("\n", " <eos> ").split()


def build_vocab(words: list[str]) -> dict[str, int]:
    """Build word-to-id mapping from word list.

    Words sorted by frequency (descending), then alphabetically for ties.
    Most frequent words get lower IDs.

    Args:
        words: List of words (with repeats)

    Returns:
        Dictionary mapping word strings to integer IDs
    """
    counter = collections.Counter(words)
    # Sort by frequency (descending), then alphabetically for determinism
    sorted_words = sorted(counter.keys(), key=lambda w: (-counter[w], w))
    return {word: i for i, word in enumerate(sorted_words)}


def ptb_producer(
    raw_data: list[int], batch_size: int, num_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate batches for stateful RNN training.

    Divides corpus into batch_size parallel streams, then yields
    consecutive (x, y) pairs. Hidden state flows across batches.

    Args:
        raw_data: 1D array of word IDs
        batch_size: Number of parallel streams
        num_steps: Sequence length per batch

    Yields:
        (x, y) tuples of shape (batch_size, num_steps)
        where y = x shifted right by 1
    """
    if len(raw_data) == 0:
        return
    
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    
    if batch_len == 0:
        return

    # Reshape into batch_size parallel streams (truncate to fit)
    data = raw_data[: batch_size * batch_len].reshape(batch_size, batch_len)

    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        x = data[:, i * num_steps : (i + 1) * num_steps]
        y = data[:, i * num_steps + 1 : (i + 1) * num_steps + 1]
        yield x, y


def get_datasets(
    batch_size: int = 20,
    num_steps: int = 35,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]:
    """Load PTB with stateful batching for language modeling.

    Returns datasets where each batch is:
    - x: (batch_size, num_steps) int32 word IDs
    - y: (batch_size, num_steps) int32 target word IDs (x shifted by 1)

    IMPORTANT: Batches must be consumed in order for stateful RNN.
    Do NOT shuffle these datasets.

    Args:
        batch_size: Number of parallel streams
        num_steps: Sequence length per batch

    Returns:
        (train_dataset, val_dataset, test_dataset, vocab_size)
    """
    # Download and build vocabulary from training data
    data_dir = download_ptb()
    train_words = read_words(os.path.join(data_dir, "ptb.train.txt"))
    word_to_id = build_vocab(train_words)
    vocab_size = len(word_to_id)

    def words_to_ids(words: list[str]) -> list[int]:
        return [word_to_id.get(w, word_to_id["<unk>"]) for w in words]

    train_ids = words_to_ids(train_words)
    val_ids = words_to_ids(read_words(os.path.join(data_dir, "ptb.valid.txt")))
    test_ids = words_to_ids(read_words(os.path.join(data_dir, "ptb.test.txt")))

    def make_dataset(data: list[int]) -> tf.data.Dataset:
        """Create tf.data.Dataset from generator."""

        def gen():
            yield from ptb_producer(data, batch_size, num_steps)

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, num_steps), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_size, num_steps), dtype=tf.int32),
            ),
        ).prefetch(tf.data.AUTOTUNE)

    return (
        make_dataset(train_ids),
        make_dataset(val_ids),
        make_dataset(test_ids),
        vocab_size,
    )
