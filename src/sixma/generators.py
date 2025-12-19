import random
import string
import inspect
from types import SimpleNamespace
from datetime import date, datetime, timedelta
from typing import Type, Any, Callable, Optional


# --- Helper: Smart Sampling ---
def smart_sample(
    edge_cases: list | set, random_fn: Callable[[], Any], rng: Any
) -> Any:
    """10% chance to pick an edge case, otherwise random."""
    if edge_cases and rng.random() < 0.10:
        return rng.choice(list(edge_cases))
    return random_fn()


class BaseGenerator:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng

    @property
    def _rng(self):
        return self.rng if self.rng is not None else random

    def bind(self, rng: random.Random):
        raise NotImplementedError("Generators must implement bind(rng)")


# --- Primitives ---


class Integer(BaseGenerator):
    def __init__(self, low: int, high: int, rng: Optional[random.Random] = None):
        super().__init__(rng)
        self.low = low
        self.high = high
        # Cache edge cases
        self._edges = {low, high, 0, 1, -1}
        self._edges = [x for x in self._edges if low <= x <= high]

    def bind(self, rng: random.Random):
        return Integer(self.low, self.high, rng=rng)

    def __iter__(self):
        # 1. Yield all edge cases sequentially
        for x in self._edges:
            yield x
        # 2. Infinite Random
        while True:
            yield self.sample()

    def sample(self) -> int:
        return smart_sample(
            self._edges,
            lambda: self._rng.randint(self.low, self.high),
            self._rng
        )


class Float(BaseGenerator):
    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        allow_nan=False,
        allow_inf=False,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(rng)
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

    def bind(self, rng: random.Random):
        return Float(
            self.low,
            self.high,
            self.allow_nan,
            self.allow_inf,
            rng=rng
        )

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> float:
        return smart_sample(
            self._edges,
            lambda: self._rng.uniform(self.low, self.high),
            self._rng
        )


class Bool(BaseGenerator):
    def bind(self, rng: random.Random):
        return Bool(rng=rng)

    def __iter__(self):
        yield False
        yield True
        while True:
            yield self.sample()

    def sample(self) -> bool:
        return self._rng.choice([True, False])


class String(BaseGenerator):
    def __init__(
        self,
        max_len: int = 20,
        chars: str = string.ascii_letters,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(rng)
        self.max_len = max_len
        self.chars = chars
        self._edges = []
        if max_len >= 0:
            self._edges.append("")
        if max_len >= 1:
            self._edges.append("a")
        if max_len > 0:
            self._edges.append(" " * max_len)

    def bind(self, rng: random.Random):
        return String(self.max_len, self.chars, rng=rng)

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> str:
        if self._edges and self._rng.random() < 0.1:
            return self._rng.choice(self._edges)
        length = self._rng.randint(0, self.max_len)
        return "".join(self._rng.choice(self.chars) for _ in range(length))


# --- Time Generators ---


class Date(BaseGenerator):
    def __init__(self, start: date, end: date, rng: Optional[random.Random] = None):
        super().__init__(rng)
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

    def bind(self, rng: random.Random):
        return Date(self.start, self.end, rng=rng)

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> date:
        return smart_sample(
            self._edges,
            lambda: self.start
            + timedelta(days=self._rng.randint(0, self.delta_days)),
            self._rng
        )


class DateTime(BaseGenerator):
    def __init__(
        self, start: datetime, end: datetime, rng: Optional[random.Random] = None
    ):
        super().__init__(rng)
        self.start = start
        self.end = end
        self.delta_seconds = int((end - start).total_seconds())
        self._edges = [start, end]

        now = datetime.now()
        if start <= now <= end:
            self._edges.append(now)

    def bind(self, rng: random.Random):
        return DateTime(self.start, self.end, rng=rng)

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> datetime:
        return smart_sample(
            self._edges,
            lambda: self.start
            + timedelta(seconds=self._rng.randint(0, self.delta_seconds)),
            self._rng
        )


# --- Combinators ---


class List(BaseGenerator):  # Removed [T] for runtime compatibility if needed
    def __init__(
        self,
        element_gen: Any,
        min_len: int = 0,
        max_len: int = 10,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(rng)
        self.element_gen = element_gen
        self.min_len = min_len
        self.max_len = max_len

    def bind(self, rng: random.Random):
        # Recursively bind the element generator if possible
        new_elem = self.element_gen
        if hasattr(new_elem, "bind"):
            new_elem = new_elem.bind(rng)
        return List(new_elem, self.min_len, self.max_len, rng=rng)

    def __iter__(self):
        # Iteration Mode: Respect the element stream
        if isinstance(self.element_gen, type):
            stream = iter(self.element_gen())
        else:
            stream = iter(self.element_gen)

        if self.min_len == 0:
            yield []

        while True:
            length = self._rng.randint(self.min_len, self.max_len)
            if length == 0 and self.min_len > 0:
                length = self.min_len
            try:
                yield [next(stream) for _ in range(length)]
            except StopIteration:
                return

    def sample(self) -> list:
        # Sampling Mode: Random access
        length = self._rng.randint(self.min_len, self.max_len)
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


class Dict(BaseGenerator):
    def __init__(self, rng: Optional[random.Random] = None, **field_generators):
        super().__init__(rng)
        self.field_gens = field_generators

    def bind(self, rng: random.Random):
        # Recursively bind fields
        bound_fields = {}
        for k, v in self.field_gens.items():
            if hasattr(v, "bind"):
                bound_fields[k] = v.bind(rng)
            else:
                bound_fields[k] = v
        return Dict(rng=rng, **bound_fields)

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


class Object(BaseGenerator):
    def __init__(
        self,
        cls: Type,
        rng: Optional[random.Random] = None,
        **field_generators
    ):
        super().__init__(rng)
        self.cls = cls
        # Bind the internal dict generator
        self.dict_gen = Dict(rng=rng, **field_generators)

    def bind(self, rng: random.Random):
        # We rely on Dict.bind to handle the fields, but we need to reconstruct Object
        # Extract fields from the internal dict_gen
        return Object(self.cls, rng=rng, **self.dict_gen.field_gens)

    def __iter__(self):
        for data in self.dict_gen:
            yield self.cls(**data)

    def sample(self) -> Any:
        data = self.dict_gen.sample()
        return self.cls(**data)


# --- Conditional / Dependent Generator ---


class Case(BaseGenerator):
    def __init__(self, rng: Optional[random.Random] = None, **steps):
        super().__init__(rng)
        self.steps = steps

    def bind(self, rng: random.Random):
        bound_steps = {}
        for k, v in self.steps.items():
            if hasattr(v, "bind"):
                bound_steps[k] = v.bind(rng)
            else:
                bound_steps[k] = v
        return Case(rng=rng, **bound_steps)

    def __iter__(self):
        keys = list(self.steps.keys())
        driver_name = keys[0]
        driver_def = self.steps[driver_name]

        if isinstance(driver_def, type):
            driver_def = driver_def()

        # Ensure driver uses our RNG if possible/applicable
        # Note: We assume it was bound during Case.bind(),
        # but if it's a fresh class instance, we might need to bind it here.
        if hasattr(driver_def, "bind") and self.rng:
             driver_def = driver_def.bind(self.rng)

        driver_stream = iter(driver_def)

        while True:
            result = {}

            # 1. Drive the primary stream
            try:
                result[driver_name] = next(driver_stream)
            except StopIteration:
                return

            # 2. Resolve Dependents
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

        if hasattr(driver_def, "bind") and self.rng:
             driver_def = driver_def.bind(self.rng)

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

            # Crucial: Bind the dynamically created generator to our RNG
            if hasattr(actual_gen, "bind") and self.rng:
                actual_gen = actual_gen.bind(self.rng)

            if hasattr(actual_gen, "sample"):
                result_dict[name] = actual_gen.sample()
            else:
                result_dict[name] = next(iter(actual_gen))
