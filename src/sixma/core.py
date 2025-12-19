import math
import functools
import inspect
import os
import random
from typing import get_type_hints, Annotated, get_args, get_origin


class PreconditionError(Exception):
    """Raised when a sample is invalid (skip this trial)."""

    pass


class CertificationError(Exception):
    """Raised when reliability criteria cannot be met (too many skips)."""

    pass


def require(condition: bool):
    """
    Asserts a precondition.
    If False, aborts the current trial and samples new inputs.
    """
    if not condition:
        raise PreconditionError()


def certify(reliability: float = 0.999, confidence: float = 0.95, max_discards: int = 10000):
    if reliability >= 1.0 or reliability <= 0.0:
        raise ValueError("Reliability must be between 0.0 and 1.0 (exclusive).")

    required_successes = math.ceil(math.log(1 - confidence) / math.log(reliability))

    def decorator(test_func):
        sig = inspect.signature(test_func)
        generator_blueprints = {}
        sixma_param_names = set()

        # STRATEGY 1: standard get_type_hints (Works for global functions)
        try:
            hints = get_type_hints(test_func, include_extras=True)
        except Exception:
            # Fallback for local functions where get_type_hints might fail
            hints = test_func.__annotations__

        # STRATEGY 2: Merge raw annotations if get_type_hints missed some
        # (Crucial for local functions like spy_test)
        combined_hints = {**test_func.__annotations__, **hints}

        for name, type_hint in combined_hints.items():
            # Check 1: Is it Annotated? (Using get_origin is safer)
            if get_origin(type_hint) is Annotated:
                args = get_args(type_hint)
                # Check 2: Is the second arg a Generator?
                if len(args) > 1:
                    gen_obj = args[1]
                    if hasattr(gen_obj, '__iter__') or isinstance(gen_obj, type):
                        generator_blueprints[name] = gen_obj
                        sixma_param_names.add(name)

        @functools.wraps(test_func)
        def wrapper(**fixture_kwargs):
            env_seed = os.environ.get("SIXMA_SEED")
            if env_seed:
                current_seed = int(env_seed)
                print(f"[Sixma] Reproducing with Seed: {current_seed}")
            else:
                # Generate a random 32-bit seed
                current_seed = random.getrandbits(32)

            # Apply the seed globally for this test execution
            random.seed(current_seed)

            successes = 0
            discards = 0

            # Setup Iterators
            active_streams = {}
            for name, blueprint in generator_blueprints.items():
                if isinstance(blueprint, type):
                    active_streams[name] = iter(blueprint())
                else:
                    active_streams[name] = iter(blueprint)

            print(f"\n[Sixma] Target: {required_successes} successes (R={reliability}, C={confidence})")

            while successes < required_successes:
                if discards > max_discards:
                    raise CertificationError(f"Discarded {discards} inputs.")

                # Generate Inputs
                generated_kwargs = {}
                try:
                    for name, stream in active_streams.items():
                        generated_kwargs[name] = next(stream)
                except StopIteration:
                     raise RuntimeError(f"Generator for '{name}' exhausted.")

                # Merge & Execute
                final_kwargs = {**fixture_kwargs, **generated_kwargs}

                try:
                    test_func(**final_kwargs)
                    successes += 1
                except PreconditionError:
                    discards += 1
                    continue
                except AssertionError as e:
                    raise AssertionError(
                        f"❌ Falsified at trial {successes + 1}!\n"
                        f"   Seed: {current_seed} (Set SIXMA_SEED={current_seed} to reproduce)\n"
                        f"   Inputs: {generated_kwargs}\n"
                        f"   Error: {e}"
                    ) from e

            print(f"[Sixma] Certified ✔️  ({successes} passed)")

        # Patch Signature
        new_params = [
            p for p in sig.parameters.values()
            if p.name not in sixma_param_names
        ]
        wrapper.__signature__ = sig.replace(parameters=new_params)

        return wrapper

    return decorator