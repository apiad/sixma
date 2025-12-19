import math
import functools
import inspect
import os
import random
from typing import get_type_hints, Annotated, get_args, get_origin


# ... (PreconditionError, CertificationError, require remain unchanged) ...
class PreconditionError(Exception):
    pass


class CertificationError(Exception):
    pass


def require(condition: bool):
    if not condition:
        raise PreconditionError()


def certify(
    reliability: float = 0.999, confidence: float = 0.95, max_discards: int = 10000
):
    if reliability >= 1.0 or reliability <= 0.0:
        raise ValueError("Reliability must be between 0.0 and 1.0 (exclusive).")

    required_successes = math.ceil(math.log(1 - confidence) / math.log(reliability))

    def decorator(test_func):
        sig = inspect.signature(test_func)
        generator_blueprints = {}
        sixma_param_names = set()

        # 1. Resolve Hints (Fallback to __annotations__ for local scope/lambdas)
        try:
            hints = get_type_hints(test_func, include_extras=True)
        except Exception:
            hints = test_func.__annotations__

        # Merge in case get_type_hints missed locals
        combined_hints = {**test_func.__annotations__, **hints}

        for name, type_hint in combined_hints.items():
            blueprint = None

            # Strategy A: Annotated[T, Generator] (Formal)
            if get_origin(type_hint) is Annotated:
                args = get_args(type_hint)
                if len(args) > 1:
                    candidate = args[1]
                    # Check if it's a class or an instance with __iter__
                    if hasattr(candidate, "__iter__") or isinstance(candidate, type):
                        blueprint = candidate

            # Strategy B: Direct Generator Instance (Shortcut)
            # e.g. def test(x: gen.Integer(0, 10))
            # We check if the hint ITSELF is iterable (and not a class like 'str' or 'list')
            elif hasattr(type_hint, "__iter__") and not isinstance(type_hint, type):
                blueprint = type_hint

            # Register if valid
            if blueprint:
                generator_blueprints[name] = blueprint
                sixma_param_names.add(name)

        @functools.wraps(test_func)
        def wrapper(**fixture_kwargs):
            # ... (Seeding logic from previous step) ...
            env_seed = os.environ.get("SIXMA_SEED")
            if env_seed:
                current_seed = int(env_seed)
                print(f"[Sixma] Reproducing with Seed: {current_seed}")
            else:
                current_seed = random.getrandbits(32)
            random.seed(current_seed)

            successes = 0
            discards = 0

            # Setup Iterators
            active_streams = {}
            for name, bp in generator_blueprints.items():
                if isinstance(bp, type):
                    try:
                        active_streams[name] = iter(bp())
                    except TypeError:
                        raise TypeError(
                            f"Generator class '{bp.__name__}' needs arguments. Instantiate it in the signature."
                        )
                else:
                    active_streams[name] = iter(bp)

            print(
                f"\n[Sixma] Target: {required_successes} successes (R={reliability}, C={confidence})"
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

                # Merge
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
            p for p in sig.parameters.values() if p.name not in sixma_param_names
        ]
        wrapper.__signature__ = sig.replace(parameters=new_params)

        return wrapper

    return decorator
