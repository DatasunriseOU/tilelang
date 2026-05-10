import sys
import types
from typing import Callable
import tilelang as tl
import tilelang.language as T

def my_macro(x: T.int32) -> T.int32:
    return x * N

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
except Exception as e:
    import traceback
    traceback.print_exc()
