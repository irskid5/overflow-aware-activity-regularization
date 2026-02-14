"""Speech Commands dataset loading with MFCC preprocessing.

Loads Google Speech Commands v0.02 and extracts MFCCs for RNN input.
Audio (16kHz, 1s) → STFT → Mel Spectrogram → Log → MFCC → (frames, coeffs)
"""
import tensorflow as tf
import tensorflow_datasets as tfds

# Audio parameters
SAMPLE_RATE = 16000
AUDIO_LENGTH = 16000  # 1 second at 16kHz

# MFCC parameters
FRAME_LENGTH = 400  # 25ms at 16kHz
FRAME_STEP = 160    # 10ms at 16kHz (60% overlap)
FFT_LENGTH = 512
NUM_SPECTROGRAM_BINS = FFT_LENGTH // 2 + 1  # = 257
NUM_MEL_BINS = 80
NUM_MFCC = 40       # MFCC coefficients per frame
LOWER_FREQ = 80.0
UPPER_FREQ = 7600.0

# Computed: (16000 - 400) / 160 + 1 = 98 frames
NUM_FRAMES = 98

# Dataset parameters
NUM_CLASSES = 12  # 10 target words + "unknown" + "silence"


def _pad_or_trim(audio: tf.Tensor) -> tf.Tensor:
    """Pad or trim audio to fixed length."""
    audio = audio[:AUDIO_LENGTH]
    pad_length = AUDIO_LENGTH - tf.shape(audio)[0]
    return tf.pad(audio, [[0, pad_length]])


def _compute_mfcc_batched(audio_batch: tf.Tensor) -> tf.Tensor:
    """Compute MFCCs from batched raw audio waveforms (GPU-optimized).
    
    Args:
        audio_batch: Raw audio tensor, shape (batch, samples), float32 normalized
        
    Returns:
        MFCC tensor, shape (batch, NUM_FRAMES, NUM_MFCC)
    """
    # STFT - works on batched input
    stfts = tf.signal.stft(
        audio_batch,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    spectrograms = tf.abs(stfts)
    
    # Mel spectrogram - mel matrix is broadcast across batch
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=NUM_SPECTROGRAM_BINS,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_FREQ,
        upper_edge_hertz=UPPER_FREQ,
    )
    mel_spectrograms = tf.tensordot(spectrograms, mel_weight_matrix, 1)
    
    # Log mel spectrogram
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    
    # MFCCs (take first NUM_MFCC coefficients)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    mfccs = mfccs[..., :NUM_MFCC]
    
    return mfccs


def _normalize_audio(audio: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalize and pad audio to fixed length."""
    # Normalize to [-1, 1] - Speech Commands uses int16 format
    audio = tf.cast(audio, tf.float32) / 32768.0
    audio = _pad_or_trim(audio)
    return audio, label


def _compute_mfcc_batch_map(
    audio_batch: tf.Tensor, label_batch: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Map function for batched MFCC computation."""
    mfccs = _compute_mfcc_batched(audio_batch)
    return mfccs, label_batch


def _prepare_dataset(
    ds: tf.data.Dataset,
    batch_size: int,
    shuffle_buffer: int | None = None,
    cache: bool = True,
) -> tf.data.Dataset:
    """Apply preprocessing pipeline with GPU-optimized batched MFCC.
    
    Pipeline: normalize → shuffle → batch → MFCC (GPU) → [cache] → prefetch
    
    Args:
        ds: Raw audio dataset
        batch_size: Batch size
        shuffle_buffer: Shuffle buffer size (None = no shuffle)
        cache: Whether to cache computed MFCCs (slower first epoch, faster subsequent)
    """
    autotune = tf.data.experimental.AUTOTUNE
    
    # Normalize audio (cheap CPU op)
    ds = ds.map(_normalize_audio, num_parallel_calls=autotune)
    
    # Shuffle before batch for better randomness (if training)
    if shuffle_buffer is not None:
        ds = ds.shuffle(buffer_size=shuffle_buffer)
    
    # Batch first, then compute MFCCs on GPU (much faster)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(_compute_mfcc_batch_map, num_parallel_calls=autotune)
    
    # Cache after MFCC computation (optional - first epoch slower, subsequent faster)
    if cache:
        ds = ds.cache()
    
    return ds.prefetch(autotune)


def get_datasets(
    batch_size: int = 64,
    enlarge: bool = False,
    cache: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load and preprocess Speech Commands dataset.
    
    Uses the 12-class version: yes, no, up, down, left, right, on, off, 
    stop, go, unknown, silence.
    
    Args:
        batch_size: Batch size for all splits
        enlarge: Ignored for Speech Commands (included for DataLoader protocol compatibility)
        cache: Whether to cache MFCCs after first epoch (default True for training)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    del enlarge  # Unused for Speech Commands (no enlarged variant)
    # Load with 12-class labels (standard keyword spotting task)
    ds_train, ds_val, ds_test = tfds.load(
        "speech_commands",
        split=["train", "validation", "test"],
        as_supervised=True,
        with_info=False,
    )
    
    ds_train = _prepare_dataset(ds_train, batch_size, shuffle_buffer=10000, cache=cache)
    ds_val = _prepare_dataset(ds_val, batch_size, cache=cache)
    ds_test = _prepare_dataset(ds_test, batch_size, cache=cache)
    
    return ds_train, ds_val, ds_test
