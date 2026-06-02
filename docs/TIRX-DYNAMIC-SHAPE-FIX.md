# tirx dynamic-symbolic-shape (`T.dynamic`) lowering fix

## Summary

The tir -> tirx namespace migration introduced a regression that crashed **every**
dynamic-symbolic-shape (`T.dynamic`) kernel during `LowerDeviceKernelLaunch`, before
codegen. This blocked the DeepSeek-V3.2 sparse-MLA forward kernel and any other
dynamic-shape kernel.

The failing assertion:

```
Check failed: (new_data_expr->IsInstance<VarNode>()) is false:
Buffer main_A_shape uses backing allocation main_A_shape, which was substituted
into the expression T.tvm_struct_get(...) and the backing allocation must be a tirx::Var
```

at `3rdparty/tvm/src/tirx/ir/stmt_functor.cc:694` (`IRSubstitute::GetRemappedBuffer`).

## Root cause

`DeviceInfoCollector` (in `LowerDeviceKernelLaunch`) walks **every** PrimFunc, including
the host wrapper produced by `MakePackedAPI`. Its `VisitStmt_(BindNode)` records each
`Bind` into `bind_map_` and runs `Substitute(value, bind_map_)` to inline locally-bound
(CSE/LICM) scalars so that launch geometry (thread extents, dynamic-shared-memory size)
is expressible from function parameters.

For a `T.dynamic` kernel, the MakePackedAPI host wrapper contains:

```python
main_A_shape   : handle = T.tvm_struct_get(A_handle, 0, 2, "handle")   # DLTensor shape array ptr
main_A_shape_1 = T.decl_buffer((2,), "int64", data=main_A_shape)       # buffer over that ptr
...
seq            : int32  = T.Cast("int32", main_A_shape_1[0])           # dynamic dim = shape[0]
```

`DeviceInfoCollector` adds the **handle-typed** binding
`main_A_shape -> tvm_struct_get(A_handle, ...)` to `bind_map_`. When it then processes the
scalar bind `seq = main_A_shape_1[0]`, it calls `Substitute(value, bind_map_)`. That value
is a `BufferLoad` over the buffer `main_A_shape_1`, whose backing-allocation `data` Var is
`main_A_shape`.

The tirx `IRSubstitute` (added in commit `81e5bcd674`, mirroring upstream apache/tvm's
`tir::Substitute`) remaps a buffer's backing `data` Var via `GetRemappedBuffer`. Because
`main_A_shape` is in the substitution map -> a non-Var `tvm_struct_get` expression, the
buffer's `data` is rewritten into that expression, and the `IsInstance<VarNode>()` ICHECK
(which correctly requires a buffer's backing allocation to remain a Var) fires.

In short: a buffer-backing **pointer** Var was being substituted with a host shape-handle
expression, then a later (correct) check required the backing allocation to be a Var.

## The fix

Launch geometry is always integer/float **scalar**. Buffer backing pointers (DLTensor
data/shape/stride handles) are never part of launch geometry. So in
`DeviceInfoCollector::VisitStmt_(BindNode)` we exclude **handle-typed** binds from
`bind_map_`:

```cpp
if (!op->var.dtype().is_handle()) {
  PrimExpr value = bind_map_.size() ? Substitute(op->value, bind_map_) : op->value;
  bind_map_.Set(op->var, value);
}
```

This keeps the scalar symbolic-shape vars (`seq`, `batch`, ...) — which ARE loaded from the
shape array and ARE used in thread extents — fully intact, while never asking `Substitute`
to rewrite a buffer's backing pointer into a non-Var expression. It is a root-cause fix: it
removes the wrong substitution, not the (correct) assertion.

Applied identically in both copies of the pass:

- `src/transform/lower_device_kernel_launch.cc` (TileLang `tl.LowerDeviceKernelLaunch`, the
  one actually used by the CUDA pipeline)
- `3rdparty/tvm/src/tirx/transform/lower_device_kernel_launch.cc`
  (`tirx.LowerDeviceKernelLaunch`, kept in sync)

## Verification (gb10, sm_121 GB10)

Rebuilt `tvm_compiler` + `tilelang` and ran:

- Minimal `T.dynamic("seq")` copy kernel: lowers to CUDA codegen (was crashing) and runs
  **correctly on GPU** at `seq=96` and `seq=160` from a single compiled kernel.
- Static-shape control kernel: still lowers to codegen (no regression).
- issue-1237 dynamic-copy-extent pattern: dynamic + static both reach codegen.

Note: the `examples/deepseek_v32/sparse_mla_fwd.py` example has a **separate, pre-existing**
failure in the experimental eager prim_func builder (`inspect.signature`/`getsource` under
cpython-3.13), unrelated to this tirx fix; the tirx `GetRemappedBuffer` ICHECK no longer
blocks the dynamic-shape codegen path.

## Precedent

Same class of tir -> tirx migration bug as tilelang `c0a6fe5a`
(`fix(loop-unswitching): alpha-rename ALL inner bound vars when cloning else arm`), where a
tirx invariant (a Var must stay a Var / be uniquely bound) was violated by a substitution
that the migration left in place.
