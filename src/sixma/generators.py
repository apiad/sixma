import random
import string
import inspect
from types import SimpleNamespace
from datetime import date, datetime, timedelta
from typing import (
    Type,
    Any,
    Callable,
    Optional,
    TypeVar,
    List as PyList,
    Dict as PyDict,
    cast,
)

T = TypeVar("T")


# --- Helper: Smart Sampling ---
def smart_sample(edge_cases: list | set, random_fn: Callable[[], Any], rng: Any) -> Any:
    """10% chance to pick an edge case, otherwise random."""
    if edge_cases and rng.random() < 0.10:
        return rng.choice(list(edge_cases))
    return random_fn()


class BaseGenerator:
    """Base class for all Sixma generators."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng

    @property
    def _rng(self):
        return self.rng if self.rng is not None else random

    def bind(self, rng: random.Random) -> "BaseGenerator":
        raise NotImplementedError("Generators must implement bind(rng)")

    def sample(self) -> Any:
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


# --- Internal Implementations ---


class _Integer(BaseGenerator):
    def __init__(self, low: int, high: int, rng: Optional[random.Random] = None):
        super().__init__(rng)
        self.low = low
        self.high = high
        self._edges = {low, high, 0, 1, -1}
        self._edges = [x for x in self._edges if low <= x <= high]

    def bind(self, rng: random.Random):
        return _Integer(self.low, self.high, rng=rng)

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> int:
        return smart_sample(
            self._edges,
            lambda: self._rng.randint(self.low, self.high),
            self._rng,
        )


class _Float(BaseGenerator):
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
        return _Float(self.low, self.high, self.allow_nan, self.allow_inf, rng=rng)

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> float:
        return smart_sample(
            self._edges,
            lambda: self._rng.uniform(self.low, self.high),
            self._rng,
        )


class _Bool(BaseGenerator):
    def bind(self, rng: random.Random):
        return _Bool(rng=rng)

    def __iter__(self):
        yield False
        yield True
        while True:
            yield self.sample()

    def sample(self) -> bool:
        return self._rng.choice([True, False])


class _String(BaseGenerator):
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
        return _String(self.max_len, self.chars, rng=rng)

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


class _Date(BaseGenerator):
    def __init__(self, start: date, end: date, rng: Optional[random.Random] = None):
        super().__init__(rng)
        self.start = start
        self.end = end
        self.delta_days = (end - start).days
        self._edges = [start, end]
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
        return _Date(self.start, self.end, rng=rng)

    def __iter__(self):
        for x in self._edges:
            yield x
        while True:
            yield self.sample()

    def sample(self) -> date:
        return smart_sample(
            self._edges,
            lambda: self.start + timedelta(days=self._rng.randint(0, self.delta_days)),
            self._rng,
        )


class _DateTime(BaseGenerator):
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
        return _DateTime(self.start, self.end, rng=rng)

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
            self._rng,
        )


class _List(BaseGenerator):
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
        new_elem = self.element_gen
        if hasattr(new_elem, "bind"):
            new_elem = new_elem.bind(rng)
        return _List(new_elem, self.min_len, self.max_len, rng=rng)

    def __iter__(self):
        # Resolve generator
        if isinstance(self.element_gen, type):
            try:
                stream = iter(self.element_gen(rng=self.rng))
            except TypeError:
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
        length = self._rng.randint(self.min_len, self.max_len)
        result = []

        # Instantiate
        if isinstance(self.element_gen, type):
            try:
                gen_obj = self.element_gen(rng=self.rng)
            except TypeError:
                gen_obj = self.element_gen()
        else:
            gen_obj = self.element_gen

        for _ in range(length):
            if hasattr(gen_obj, "sample"):
                result.append(gen_obj.sample())
            else:
                result.append(next(iter(gen_obj)))
        return result


class _Dict(BaseGenerator):
    def __init__(self, rng: Optional[random.Random] = None, **field_generators):
        super().__init__(rng)
        self.field_gens = field_generators

    def bind(self, rng: random.Random):
        bound_fields = {}
        for k, v in self.field_gens.items():
            if hasattr(v, "bind"):
                bound_fields[k] = v.bind(rng)
            else:
                bound_fields[k] = v
        return _Dict(rng=rng, **bound_fields)

    def __iter__(self):
        streams = {}
        for k, gen in self.field_gens.items():
            if isinstance(gen, type):
                try:
                    streams[k] = iter(gen(rng=self.rng))
                except TypeError:
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
            if isinstance(gen, type):
                try:
                    gen_obj = gen(rng=self.rng)
                except TypeError:
                    gen_obj = gen()
            else:
                gen_obj = gen

            if hasattr(gen_obj, "sample"):
                result[k] = gen_obj.sample()
            else:
                result[k] = next(iter(gen_obj))
        return result


class _Object(BaseGenerator):
    def __init__(
        self, cls: Type, rng: Optional[random.Random] = None, **field_generators
    ):
        super().__init__(rng)
        self.cls = cls
        self.dict_gen = _Dict(rng=rng, **field_generators)

    def bind(self, rng: random.Random):
        bound_dict = self.dict_gen.bind(rng)
        return _Object(self.cls, rng=rng, **bound_dict.field_gens)

    def __iter__(self):
        for data in self.dict_gen:
            yield self.cls(**data)

    def sample(self) -> Any:
        data = self.dict_gen.sample()
        return self.cls(**data)


class _Case(BaseGenerator):
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
        return _Case(rng=rng, **bound_steps)

    def __iter__(self):
        keys = list(self.steps.keys())
        driver_name = keys[0]
        driver_def = self.steps[driver_name]

        if isinstance(driver_def, type):
            try:
                driver_def = driver_def(rng=self.rng)
            except TypeError:
                driver_def = driver_def()
        elif hasattr(driver_def, "bind") and self.rng:
            driver_def = driver_def.bind(self.rng)

        driver_stream = iter(driver_def)

        while True:
            result = {}
            try:
                result[driver_name] = next(driver_stream)
            except StopIteration:
                return
            self._resolve_dependents(keys[1:], result)
            yield SimpleNamespace(**result)

    def sample(self) -> SimpleNamespace:
        keys = list(self.steps.keys())
        result = {}
        driver_name = keys[0]
        driver_def = self.steps[driver_name]

        if isinstance(driver_def, type):
            try:
                driver_def = driver_def(rng=self.rng)
            except TypeError:
                driver_def = driver_def()
        elif hasattr(driver_def, "bind") and self.rng:
            driver_def = driver_def.bind(self.rng)

        if hasattr(driver_def, "sample"):
            result[driver_name] = driver_def.sample()
        else:
            result[driver_name] = next(iter(driver_def))

        self._resolve_dependents(keys[1:], result)
        return SimpleNamespace(**result)

    def _resolve_dependents(self, dependent_keys, result_dict):
        for name in dependent_keys:
            step_def = self.steps[name]
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
                try:
                    actual_gen = actual_gen(rng=self.rng)
                except TypeError:
                    actual_gen = actual_gen()
            elif hasattr(actual_gen, "bind") and self.rng:
                actual_gen = actual_gen.bind(self.rng)

            if hasattr(actual_gen, "sample"):
                result_dict[name] = actual_gen.sample()
            else:
                result_dict[name] = next(iter(actual_gen))


# --- Public Factory Functions (Typed for Mypy) ---


def Integer(low: int, high: int) -> int:
    """Returns an integer generator. Mypy treats this as an int."""
    return cast(int, _Integer(low, high))


def Float(
    low: float = 0.0,
    high: float = 1.0,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> float:
    """Returns a float generator. Mypy treats this as a float."""
    return cast(float, _Float(low, high, allow_nan, allow_inf))


def Bool() -> bool:
    """Returns a bool generator. Mypy treats this as a bool."""
    return cast(bool, _Bool())


def String(max_len: int = 20, chars: str = string.ascii_letters) -> str:
    """Returns a string generator. Mypy treats this as a str."""
    return cast(str, _String(max_len, chars))


def Date(start: date, end: date) -> date:
    """Returns a date generator. Mypy treats this as a date."""
    return cast(date, _Date(start, end))


def DateTime(start: datetime, end: datetime) -> datetime:
    """Returns a datetime generator. Mypy treats this as a datetime."""
    return cast(datetime, _DateTime(start, end))


def List(element_gen: T | Any, min_len: int = 0, max_len: int = 10) -> PyList[T]:
    """Returns a list generator. Mypy treats this as a List[T]."""
    return cast(PyList[T], _List(element_gen, min_len, max_len))


def Dict(**field_generators: Any) -> PyDict[str, Any]:
    """Returns a dict generator. Mypy treats this as a Dict."""
    return cast(PyDict[str, Any], _Dict(**field_generators))


def Object(cls: Type[T], **field_generators: Any) -> T:
    """Returns an object generator. Mypy treats this as an instance of T."""
    return cast(T, _Object(cls, **field_generators))


def Case(**steps: Any) -> SimpleNamespace:
    """Returns a Case generator. Mypy treats this as a SimpleNamespace."""
    return cast(SimpleNamespace, _Case(**steps))
