from aria_dataset.aria_dataset_dataset_builder import AriaDataset

# register with tfds when this module is imported
try:
    import tensorflow_datasets as tfds
    # tfds 4.0+ uses register, older versions auto-register on import
    if hasattr(tfds, 'core') and hasattr(tfds.core, 'registered'):
        tfds.core.registered.register(AriaDataset)
except Exception:
    pass  # silently fail if tfds not available or registration fails

__all__ = ['AriaDataset']
