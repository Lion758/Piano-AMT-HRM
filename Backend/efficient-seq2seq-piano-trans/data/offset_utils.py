from collections.abc import Mapping


def _config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)

    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass

    return getattr(config, key, default)


def resolve_use_truth_offsets(config) -> bool:
    """Resolve raw-offset mode with V4's note-extension toggle taking precedence."""
    data_config = _config_get(config, "data", config)
    use_note_extensions = _config_get(data_config, "use_note_extensions", None)
    if use_note_extensions is not None:
        return not bool(use_note_extensions)

    return bool(_config_get(data_config, "use_truth_offsets", False))
