import math
import functools
import inspect
import os
import random
from typing import get_type_hints, Annotated, get_args, get_origin


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

        # 1. Resolve Hints
        try:
            hints = get_type_hints(test_func, include_extras=True)
        except Exception:
            hints = test_func.__annotations__

        combined_hints = {**test_func.__annotations__, **hints}

        for name, type_hint in combined_hints.items():
            blueprint = None

            # Strategy A: Annotated[T, Generator]
            if get_origin(type_hint) is Annotated:
                args = get_args(type_hint)
                if len(args) > 1:
                    candidate = args[1]
                    if hasattr(candidate, "__iter__") or isinstance(candidate, type):
                        blueprint = candidate

            # Strategy B: Direct Generator Instance
            elif hasattr(type_hint, "__iter__") and not isinstance(type_hint, type):
                blueprint = type_hint

            if blueprint:
                generator_blueprints[name] = blueprint
                sixma_param_names.add(name)

        @functools.wraps(test_func)
        def wrapper(**fixture_kwargs):
            # 1. SEEDING (Thread-Safe)
            env_seed = os.environ.get("SIXMA_SEED")
            if env_seed:
                current_seed = int(env_seed)
                print(f"[Sixma] Reproducing with Seed: {current_seed}")
            else:
                # Generate a random seed for this run
                # Using os.urandom or time ensures it's unique per call/thread
                current_seed = random.getrandbits(32)

            # Create a LOCAL Random instance. DO NOT use random.seed() globally.
            rng = random.Random(current_seed)

            successes = 0
            discards = 0

            # 2. Setup Iterators (Bound to local RNG)
            active_streams = {}
            for name, bp in generator_blueprints.items():
                if isinstance(bp, type):
                    # If it's a class, instantiate it.
                    # If it supports 'rng' in init (our Generators), pass it.
                    try:
                        instance = bp(rng=rng)
                    except TypeError:
                        # Fallback for custom classes that don't take rng
                        instance = bp()
                    active_streams[name] = iter(instance)
                else:
                    # If it's an instance, try to BIND it to our RNG
                    if hasattr(bp, "bind"):
                        bound_bp = bp.bind(rng)
                        active_streams[name] = iter(bound_bp)
                    else:
                        # Fallback: iterate as is (might use global random)
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

        new_params = [
            p for p in sig.parameters.values() if p.name not in sixma_param_names
        ]
        wrapper.__signature__ = sig.replace(parameters=new_params)

        return wrapper

    return decorator
