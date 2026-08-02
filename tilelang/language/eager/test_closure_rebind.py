import types
import tilelang.language as T


def my_macro(x: T.int32) -> T.int32:
    # ``rebind_macro`` supplies this global before the function is called.
    return x * N  # noqa: F821


# Rebind closure
def rebind_macro(func, extra_globals):
    new_globals = {"__name__": func.__module__, "__builtins__": __builtins__}
    new_globals.update(extra_globals)
    return types.FunctionType(
        func.__code__,
        new_globals,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )


new_macro = rebind_macro(my_macro, {"N": 2})

try:
    wrapped = T.macro(new_macro)
    print("Success!")
except Exception:
    import traceback

    traceback.print_exc()
