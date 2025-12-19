import random
import string
from typing import Type

# --- Primitives ---


class Integer:
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def __iter__(self):
        # 1. Edge Cases (Bounds + Zero + One)
        candidates = {self.low, self.high, 0, 1, -1}
        for x in sorted(candidates):
            if self.low <= x <= self.high:
                yield x

        # 2. Infinite Random
        while True:
            yield random.randint(self.low, self.high)


class Float:
    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        allow_nan: bool = False,
        allow_inf: bool = False,
    ):
        self.low = low
        self.high = high
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf

    def __iter__(self):
        # 1. Edge Cases
        # Note: We filter duplicates, but nan != nan so handle it separately
        candidates = {self.low, self.high, 0.0, -1.0, 1.0}

        # Yield standard numeric edge cases if in range
        for x in sorted(candidates):
            if self.low <= x <= self.high:
                yield x

        # Special non-numeric edge cases
        if self.allow_inf:
            yield float("inf")
            yield float("-inf")
        if self.allow_nan:
            yield float("nan")

        # 2. Infinite Random
        while True:
            yield random.uniform(self.low, self.high)


class Bool:
    def __iter__(self):
        # 1. Edge Cases (Both)
        yield False
        yield True

        # 2. Infinite Random
        while True:
            yield random.choice([True, False])


class String:
    def __init__(self, max_len: int = 20, chars: str = string.ascii_letters):
        self.max_len = max_len
        self.chars = chars

    def __iter__(self):
        # 1. Edge Cases
        if self.max_len >= 0:
            yield ""
        if self.max_len >= 1:
            yield "a"
        if self.max_len > 0:
            yield " " * self.max_len

        # 2. Infinite Random
        while True:
            length = random.randint(0, self.max_len)
            yield "".join(random.choice(self.chars) for _ in range(length))


# --- Combinators ---


class List[T]:
    """
    Generates lists of type T.
    The element_gen is used to populate the lists.
    """

    def __init__(self, element_gen: T, min_len: int = 0, max_len: int = 10):
        self.element_gen = element_gen
        self.min_len = min_len
        self.max_len = max_len

    def __iter__(self):
        # Create a single stream for elements.
        # This means the first list generated will contain the 'edge cases'
        # of the element generator.
        if isinstance(self.element_gen, type):
            stream = iter(self.element_gen())
        else:
            stream = iter(self.element_gen)

        # 1. Edge Case: Empty List
        if self.min_len == 0:
            yield []

        # 2. Infinite Random Lists
        while True:
            length = random.randint(self.min_len, self.max_len)
            if length == 0 and self.min_len > 0:
                length = self.min_len

            # Pull 'length' items from the element stream
            # We use try/next in case the element generator is finite (rare but possible)
            try:
                yield [next(stream) for _ in range(length)]
            except StopIteration:
                # If element generator runs out, we must stop too
                return


class Dict[T]:
    """
    Generates dictionaries with a fixed schema.
    Usage: Dict(name=String(), age=Integer(0,100))
    """

    def __init__(self, **field_generators):
        self.field_gens = field_generators

    def __iter__(self):
        # Create streams for all fields
        streams = {}
        for k, gen in self.field_gens.items():
            if isinstance(gen, type):
                streams[k] = iter(gen())
            else:
                streams[k] = iter(gen)

        # Infinite Loop
        while True:
            try:
                # Construct dict by pulling one item from each stream
                yield {k: next(s) for k, s in streams.items()}
            except StopIteration:
                return


class Object[T]:
    """
    Generates instances of class 'cls' using kwargs.
    Usage: Object(User, id=Integer(0,10), name=String())
    """

    def __init__(self, cls: Type[T], **field_generators):
        self.cls = cls
        self.field_gens = field_generators

    def __iter__(self):
        # Reuse logic from Dict to get the kwargs
        dict_gen = Dict(**self.field_gens)

        for kwargs in dict_gen:
            yield self.cls(**kwargs)
