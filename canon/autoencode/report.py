import logging
import keras

__logger = logging.getLogger(__name__)

def backend_report():
    """Print a concise report of Keras backend and GPU state.

    Covers both the *torch* and *tensorflow* backends and handles
    CPU‑only situations gracefully.
    """
    backend = keras.backend.backend()
    __logger.info("―" * 72)
    __logger.info(f"Keras backend       : {backend}")

    if backend == "torch":
        import torch
        __logger.info(f"PyTorch version     : {torch.__version__}")
        __logger.info(f"CUDA available      : {torch.cuda.is_available()}")
        __logger.info(f"Device count        : {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            dev_id = torch.cuda.current_device()
            __logger.info(f"Active GPU id       : {dev_id}")
            __logger.info(f"Active GPU name     : {torch.cuda.get_device_name(dev_id)}")
        else:
            __logger.info("Running on CPU; GPU not detected.")

    elif backend == "tensorflow":
        import tensorflow as tf
        __logger.info(f"TensorFlow version  : {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        __logger.info(f"CUDA built‑with     : {tf.test.is_built_with_cuda()}")
        __logger.info(f"Visible GPUs        : {len(gpus)}")
        for g in gpus:
            __logger.info(f"  • {g}")
        if not gpus:
            __logger.info("Running on CPU; no GPUs detected by TensorFlow.")
    else:
        __logger.info("Unknown backend; cannot report GPU status.")
    __logger.info("―" * 72)

def is_using_gpu():
    """Check if GPU is being used by the current Keras backend.
    
    Returns:
        bool: True if GPU is available and being used, False otherwise.
    """
    backend = keras.backend.backend()
    
    if backend == "torch":
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    
    elif backend == "tensorflow":
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        return len(gpus) > 0
    
    # For other backends like JAX or unknown backends
    return False

