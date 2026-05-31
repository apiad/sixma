import math
import functools
import inspect
import os
import random
import logging
from typing import get_type_hints, Annotated, get_args, get_origin
from sixma.generators import BaseGenerator

# Setup Logger
logger = logging.getLogger("sixma")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class PreconditionError(Exception):
    pass

class CertificationError(Exception):
    pass


class _Discard(Exception):
    """Internal sentinel raised by ctx.discard() — caught by the trial loop."""
    pass


class _TrialTooDeep(Exception):
    """Internal sentinel raised when a trial exceeds max_picks_per_trial."""
    pass


class Trial:
    """Per-trial state. Not user-facing."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.picks: list[tuple[str, object]] = []
        self.pick_count: int = 0


class UniformSampler:
    """Uniform random sampler. The v1 default."""

    def choose(self, options, trial: "Trial"):
        return trial.rng.choice(options)

    def randint(self, low: int, high: int, trial: "Trial") -> int:
        return trial.rng.randint(low, high)


class PickContext:
    """Test-body-facing facade. Records each decision into `trial.picks` and
    delegates the random choice to the sampler.
    """

    def __init__(self, trial: "Trial", sampler, max_picks: int = 1000):
        self._trial = trial
        self._sampler = sampler
        self._max_picks = max_picks

    @property
    def trace(self) -> list[tuple[str, object]]:
        return list(self._trial.picks)

    def _check_budget(self):
        if self._trial.pick_count >= self._max_picks:
            raise _TrialTooDeep()
        self._trial.pick_count += 1

    def _label(self, label):
        if label is None:
            return f"pick_{len(self._trial.picks)}"
        return label

    def pick(self, *options, label=None):
        if not options:
            raise ValueError("ctx.pick requires at least one option")
        self._check_budget()
        chosen = self._sampler.choose(list(options), self._trial)
        self._trial.picks.append((self._label(label), chosen))
        return chosen

    def range(self, low: int, high: int, label=None) -> int:
        self._check_budget()
        chosen = self._sampler.randint(low, high, self._trial)
        self._trial.picks.append((self._label(label), chosen))
        return chosen

    def discard(self):
        raise _Discard()


def require(condition: bool):
    if not condition:
        raise PreconditionError()

def certify(
    reliability: float = 0.999,
    confidence: float = 0.95,
    max_discards: int = 10000,
    max_picks_per_trial: int = 1000,
):
    if reliability >= 1.0 or reliability <= 0.0:
        raise ValueError("Reliability must be between 0.0 and 1.0 (exclusive).")

    required_successes = math.ceil(math.log(1 - confidence) / math.log(reliability))

    def decorator(test_func):
        sig = inspect.signature(test_func)
        generator_blueprints = {}
        sixma_param_names = set()

        # Strategy 1: Check Default Values (The Preferred "Strict Mypy" Way)
        for name, param in sig.parameters.items():
            if isinstance(param.default, BaseGenerator):
                generator_blueprints[name] = param.default
                sixma_param_names.add(name)

        # Strategy 2: Check Annotations (Fallback for "Quick Scripts")
        # Only check args we haven't found yet
        if len(generator_blueprints) < len(sig.parameters):
            try:
                hints = get_type_hints(test_func, include_extras=True)
            except Exception:
                hints = test_func.__annotations__

            combined_hints = {**test_func.__annotations__, **hints}

            for name, type_hint in combined_hints.items():
                if name in sixma_param_names:
                    continue

                blueprint = None
                if get_origin(type_hint) is Annotated:
                    args = get_args(type_hint)
                    if len(args) > 1:
                        candidate = args[1]
                        if isinstance(candidate, BaseGenerator) or (isinstance(candidate, type) and issubclass(candidate, BaseGenerator)):
                            blueprint = candidate
                # but let's keep it safe for class types
                elif hasattr(type_hint, "__iter__") and not isinstance(type_hint, type):
                     # If it's a raw instance not inheriting BaseGenerator, it's risky but allowed in v0
                     # For v1, let's enforce BaseGenerator for safety?
                     # Let's keep duck typing for iterables that MIGHT be generators but not our classes
                     blueprint = type_hint

                if blueprint:
                    generator_blueprints[name] = blueprint
                    sixma_param_names.add(name)

        # Strategy 3: Identify the ctx (PickContext) parameter — at most one.
        ctx_param_name = None
        for name, param in sig.parameters.items():
            if name in sixma_param_names:
                continue
            type_hint = test_func.__annotations__.get(name)
            is_ctx = (
                name == "ctx"
                or type_hint is PickContext
                or (isinstance(type_hint, type) and issubclass(type_hint, PickContext))
            )
            if is_ctx:
                if ctx_param_name is not None:
                    raise ValueError(
                        f"at most one ctx parameter allowed; "
                        f"found '{ctx_param_name}' and '{name}'"
                    )
                ctx_param_name = name
                sixma_param_names.add(name)

        @functools.wraps(test_func)
        def wrapper(**fixture_kwargs):
            # 1. SEEDING
            env_seed = os.environ.get("SIXMA_SEED")
            if env_seed:
                current_seed = int(env_seed)
                logger.info(f"[Sixma] Reproducing with Seed: {current_seed}")
            else:
                current_seed = random.getrandbits(32)

            rng = random.Random(current_seed)

            successes = 0
            discards = 0

            # 2. Setup Iterators
            active_streams = {}
            for name, bp in generator_blueprints.items():
                if isinstance(bp, type):
                    try:
                        instance = bp(rng=rng)
                    except TypeError:
                        instance = bp()
                    active_streams[name] = iter(instance)
                else:
                    # It's an instance (like _Integer)
                    if hasattr(bp, "bind"):
                        bound_bp = bp.bind(rng)
                        active_streams[name] = iter(bound_bp)
                    else:
                        active_streams[name] = iter(bp)

            logger.info(
                f"[Sixma] Target: {required_successes} successes (R={reliability}, C={confidence})"
            )

            while successes < required_successes:
                if discards > max_discards:
                    raise CertificationError(f"Discarded {discards} inputs.")

                # Generate
                generated_kwargs = {}
                for name, stream in active_streams.items():
                    try:
                        generated_kwargs[name] = next(stream)
                    except StopIteration:
                        raise RuntimeError(f"Generator for '{name}' exhausted.")

                # Build a PickContext for this trial if the body wants one
                trial = None
                if ctx_param_name is not None:
                    trial = Trial(rng=rng)
                    generated_kwargs[ctx_param_name] = PickContext(
                        trial=trial,
                        sampler=UniformSampler(),
                        max_picks=max_picks_per_trial,
                    )

                final_kwargs = {**fixture_kwargs, **generated_kwargs}

                try:
                    test_func(**final_kwargs)
                    successes += 1
                except PreconditionError:
                    discards += 1
                    continue
                except _Discard:
                    discards += 1
                    continue
                except _TrialTooDeep:
                    discards += 1
                    continue
                except AssertionError as e:
                    if ctx_param_name is not None and trial is not None:
                        picks = trial.picks
                        if len(picks) > 20:
                            shown = (
                                picks[:10]
                                + [("...", f"({len(picks) - 15} picks elided)")]
                                + picks[-5:]
                            )
                        else:
                            shown = picks
                        trace_lines = [f"     {label} = {value!r}" for label, value in shown]
                        trace_block = "\n".join(trace_lines) if trace_lines else "     (no picks)"
                        error_msg = (
                            f"❌ Falsified at trial {successes + 1}!\n"
                            f"   Seed: {current_seed} (Set SIXMA_SEED={current_seed} to reproduce)\n"
                            f"   Trace:\n{trace_block}\n"
                            f"   Error: {e}"
                        )
                        logger.error(error_msg)
                        raise AssertionError(error_msg) from e

                    # Legacy path: shrinking + Inputs: dict
                    minimal_msg = ""
                    try:
                        min_kwargs = {}
                        for name, bp in generator_blueprints.items():
                            gen_inst = bp() if isinstance(bp, type) else bp
                            min_kwargs[name] = next(iter(gen_inst))

                        final_min_kwargs = {**fixture_kwargs, **min_kwargs}
                        test_func(**final_min_kwargs)
                    except AssertionError:
                        minimal_msg = f"\n   📉 Minimal Counter-Example: {min_kwargs}"
                    except Exception:
                        pass

                    error_msg = (
                        f"❌ Falsified at trial {successes + 1}!\n"
                        f"   Seed: {current_seed} (Set SIXMA_SEED={current_seed} to reproduce)\n"
                        f"   Inputs: {generated_kwargs}"
                        f"{minimal_msg}\n"
                        f"   Error: {e}"
                    )
                    logger.error(error_msg)
                    raise AssertionError(error_msg) from e

            logger.info(f"[Sixma] Certified ✔️  ({successes} passed)")

        new_params = [
            p for p in sig.parameters.values() if p.name not in sixma_param_names
        ]
        wrapper.__signature__ = sig.replace(parameters=new_params)

        return wrapper

    return decorator
