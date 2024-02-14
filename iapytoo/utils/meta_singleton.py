class MetaSingleton(type):
    """
    Meta classe pour définir des singletons.

    exemple:
    class MonSingleton(metaclass=MetaSingleton)

    """

    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in MetaSingleton.__instances:
            MetaSingleton.__instances[cls] = super(MetaSingleton, cls).__call__(
                *args, **kwargs
            )
        return MetaSingleton.__instances[cls]
