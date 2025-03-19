import threading


def singleton(cls):
    """Décorateur pour rendre une classe Singleton thread-safe."""
    instances = {}
    lock = threading.RLock()  # Verrou pour synchroniser l'accès aux instances

    def get_instance(*args, **kwargs):
        nonlocal instances
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
