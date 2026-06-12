from __future__ import annotations

import logging
import os
import warnings

_TF_RUNTIME_CONFIGURED = False


def configure_tensorflow_runtime() -> None:
    """Reduce TensorFlow / oneDNN / Keras noise before first import."""
    global _TF_RUNTIME_CONFIGURED
    if _TF_RUNTIME_CONFIGURED:
        return
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*reset_default_graph.*",
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"keras.*")
    _TF_RUNTIME_CONFIGURED = True


def import_tensorflow():
    configure_tensorflow_runtime()
    import tensorflow as tf

    for logger_name in ("tensorflow", "keras", "absl"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    tf.get_logger().setLevel("ERROR")
    return tf
