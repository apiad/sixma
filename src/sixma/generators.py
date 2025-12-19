import random
import string
import inspect
from types import SimpleNamespace
from datetime import date, datetime, timedelta
from typing import Type, Any, Callable


# --- Helper: Smart Sampling ---
def smart_sample(edge_cases: list | set, random_fn: Callable[[], Any]) -> Any:
    """10% chance to pick an edge case, otherwise random."""
    if edge_cases and random.random() < 0.10:
        return random.choice(list(edge_cases))
    return random_fn()


# --- Primitives ---


class Integer:
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        # Cache edge cases
        self._edges = {low, high, 0, 1, -1}
        self._edges = [x for x in self._edges if low <= x <= high]

    def __iter__(self):
        # 1. Yield all edge cases sequentially
        for x in self._edges:
            yield x
        # 2. Infinite Random
        while True:
            yield self.sample()

    def sample(self) -> int:
        return smart_sample(self._edges, lambda: random.randint(self.low, self.high))


class Float:
    def __init__(
        self, low: float = 0.0, high: float = 1.0, allow_nan=False, allow_inf=False
    ):
        self.low = low
        self.high = high
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf

        edges = {low, high, 0.0, -1.0, 1.0}
        self._edges = [x for x in edges if low <= x <= high]
        if allow_inf:
            self._edges.extend([float("inf"), float("-inf")])
        if allow_nan:
            self._edges.append(float("nan"))

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> float:
        return smart_sample(self._edges, lambda: random.uniform(self.low, self.high))


class Bool:
    def __iter__(self):
        yield False
        yield True
        while True:
            yield self.sample()

    def sample(self) -> bool:
        return random.choice([True, False])


class String:
    def __init__(self, max_len: int = 20, chars: str = string.ascii_letters):
        self.max_len = max_len
        self.chars = chars
        self._edges = []
        if max_len >= 0:
            self._edges.append("")
        if max_len >= 1:
            self._edges.append("a")
        if max_len > 0:
            self._edges.append(" " * max_len)

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> str:
        if self._edges and random.random() < 0.1:
            return random.choice(self._edges)
        length = random.randint(0, self.max_len)
        return "".join(random.choice(self.chars) for _ in range(length))


# --- Time Generators ---


class Date:
    def __init__(self, start: date, end: date):
        self.start = start
        self.end = end
        self.delta_days = (end - start).days
        self._edges = [start, end]

        # Try to find 'today' and a leap day
        today = date.today()
        if start <= today <= end:
            self._edges.append(today)

        curr_year = start.year
        for y in range(curr_year, curr_year + 5):
            try:
                leap_day = date(y, 2, 29)
                if start <= leap_day <= end:
                    self._edges.append(leap_day)
                    break
            except ValueError:
                continue

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> date:
        return smart_sample(
            self._edges,
            lambda: self.start + timedelta(days=random.randint(0, self.delta_days)),
        )


class DateTime:
    def __init__(self, start: datetime, end: datetime):
        self.start = start
        self.end = end
        self.delta_seconds = int((end - start).total_seconds())
        self._edges = [start, end]

        now = datetime.now()
        if start <= now <= end:
            self._edges.append(now)

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> datetime:
        return smart_sample(
            self._edges,
            lambda: self.start
            + timedelta(seconds=random.randint(0, self.delta_seconds)),
        )


# --- Combinators ---


class List[T]:
    def __init__(self, element_gen: T, min_len: int = 0, max_len: int = 10):
        self.element_gen = element_gen
        self.min_len = min_len
        self.max_len = max_len

    def __iter__(self):
        # Iteration Mode: Respect the element stream
        if isinstance(self.element_gen, type):
            stream = iter(self.element_gen())
        else:
            stream = iter(self.element_gen)

        if self.min_len == 0:
            yield []

        while True:
            length = random.randint(self.min_len, self.max_len)
            if length == 0 and self.min_len > 0:
                length = self.min_len
            try:
                yield [next(stream) for _ in range(length)]
            except StopIteration:
                return

    def sample(self) -> list[T]:
        # Sampling Mode: Random access
        length = random.randint(self.min_len, self.max_len)
        result = []

        # Instantiate/Resolve generator logic
        gen_obj = (
            self.element_gen()
            if isinstance(self.element_gen, type)
            else self.element_gen
        )

        for _ in range(length):
            if hasattr(gen_obj, "sample"):
                result.append(gen_obj.sample())
            else:
                result.append(next(iter(gen_obj)))
        return result


class Dict[T]:
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

        while True:
            try:
                yield {k: next(s) for k, s in streams.items()}
            except StopIteration:
                return

    def sample(self) -> dict:
        result = {}
        for k, gen in self.field_gens.items():
            gen_obj = gen() if isinstance(gen, type) else gen
            if hasattr(gen_obj, "sample"):
                result[k] = gen_obj.sample()
            else:
                result[k] = next(iter(gen_obj))
        return result


class Object[T]:
    def __init__(self, cls: Type[T], **field_generators):
        self.cls = cls
        self.dict_gen = Dict(**field_generators)

    def __iter__(self):
        for data in self.dict_gen:
            yield self.cls(**data)

    def sample(self) -> T:
        data = self.dict_gen.sample()
        return self.cls(**data)


# --- Conditional / Dependent Generator ---


class Case:
    """
    Generates a namespace where fields depend on previous fields.
    """

    def __init__(self, **steps):
        self.steps = steps

    def __iter__(self):
        keys = list(self.steps.keys())
        driver_name = keys[0]
        driver_def = self.steps[driver_name]

        if isinstance(driver_def, type):
            driver_def = driver_def()
        driver_stream = iter(driver_def)

        while True:
            result = {}

            # 1. Drive the primary stream
            try:
                result[driver_name] = next(driver_stream)
            except StopIteration:
                return

            # 2. Resolve Dependents via Sampling
            self._resolve_dependents(keys[1:], result)

            yield SimpleNamespace(**result)

    def sample(self) -> SimpleNamespace:
        keys = list(self.steps.keys())
        result = {}

        # 1. Sample the driver
        driver_name = keys[0]
        driver_def = self.steps[driver_name]
        if isinstance(driver_def, type):
            driver_def = driver_def()

        if hasattr(driver_def, "sample"):
            result[driver_name] = driver_def.sample()
        else:
            result[driver_name] = next(iter(driver_def))

        # 2. Resolve Dependents
        self._resolve_dependents(keys[1:], result)

        return SimpleNamespace(**result)

    def _resolve_dependents(self, dependent_keys, result_dict):
        for name in dependent_keys:
            step_def = self.steps[name]

            # Check for dynamic dependency (callable lambda)
            if (
                callable(step_def)
                and not isinstance(step_def, type)
                and not hasattr(step_def, "__iter__")
            ):
                sig = inspect.signature(step_def)
                args = {k: result_dict[k] for k in sig.parameters if k in result_dict}
                actual_gen = step_def(**args)
            else:
                actual_gen = step_def

            if isinstance(actual_gen, type):
                actual_gen = actual_gen()

            if hasattr(actual_gen, "sample"):
                result_dict[name] = actual_gen.sample()
            else:
                result_dict[name] = next(iter(actual_gen))
