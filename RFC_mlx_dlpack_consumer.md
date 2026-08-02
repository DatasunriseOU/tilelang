# RFC: DLPack consumer support for MLX arrays

We have a working downstream prototype that lets MLX consume DLPack-exporting
objects:

- CPU DLPack capsules / producers.
- MLX self round-trips through mx.array(x.__dlpack__()).
- Metal-resident kDLMetal capsules where data is a foreign
  MTL::Buffer*.
- TileLang / TVM-FFI Metal tensors exported as DLPack and re-wrapped as
  mx.array without a host copy.

This is directly related to ml-explore/mlx issue #2848, where the current
behavior is that mx.array(...) accepts CPU tensors from other frameworks but
not device tensors. The maintainer comment said that accepting MPS / CUDA
arrays may be possible, while PyTorch MPS is blocked until PyTorch exports MPS
DLPack.

Issue link:

- https://github.com/ml-explore/mlx/issues/2848

Proof-of-concept branch:

- https://github.com/DatasunriseOU/mlx

Scope note: this PoC does **not** implement generic CUDA DLPack import yet. It
supports CPU and Metal. kDLCUDA is explicitly rejected today, so CUDA should
be treated as a follow-up if the MLX team wants parity with the new MLX CUDA
backend.

---

## Verified State

I re-checked the repos on 2026-05-13.

Upstream MLX:

text
ml-explore/mlx main
8f4099d8243dfbc814b07ca6b4606196f0c490ce
2026-05-12 [CUDA] Guard qmm_naive scale and bias loads at tile boundaries (#3509)

Downstream PoC:

text
DatasunriseOU/mlx main
3a6039d776fb2d331d8d8575f9f2af533a3b1680
2026-05-13 Add Python array bridge dylib

Merge-base:

text
b08ec310f1506487a32d8fdeb71cc8465c242eae
2026-05-11 Fix scatter_prod GPU hang on NaN with contention (#3492)

Ahead/behind after fetching both remotes:

text
upstream/main...origin/main = 5 behind / 4 ahead

Net downstream delta from the merge-base:

text
25 files changed, 1282 insertions(+), 24 deletions(-)

That is the measured diff for the current downstream branch; the count should
be refreshed before opening PRs because ml-explore/mlx@main is moving.

---

## Why This Matters

MLX already exports DLPack from mx.array:

- mx.array.__dlpack__
- mx.array.__dlpack_device__

But upstream MLX does not currently expose the corresponding consumer side:

- no public mx.from_dlpack.
- no mx.array(...) path that accepts a DLPack capsule or producer object.

For TileLang, the missing consumer path means the zero-copy path is only
half-duplex:

1. MLX array -> DLPack -> TVM/TileLang Metal tensor works.
2. TVM/TileLang Metal tensor -> DLPack -> MLX array requires downstream MLX
   patches.

The practical use case is simple: TileLang emits Metal kernels, TVM-FFI
executes them, and MLX owns the surrounding model graph. All tensors are on the
same Apple GPU memory system. Without a consumer path, the boundary either
forces a copy or forces framework-specific native glue.

---

## Chronology

### MLX side

The PoC branch has four downstream commits over the merge-base:

1. c0cda6e20 - 2026-05-11 - Fix mx.array DLPack dispatch
   - Adds the DLPack consumer implementation.
   - Adds CPU import and a Metal build stub.
   - 8 files, 662 insertions, 1 deletion.

2. 41ec3f5a8 - 2026-05-11 - Support DLPack Metal interop for cppmega
   - Adds the kDLMetal path.
   - Wires imported Metal-backed arrays into MLX custom-kernel / fast paths.
   - 16 files, 525 insertions, 59 deletions.
   - Before upstreaming, rename the commit to something like
     [Metal] Add DLPack consumer for MTLBuffer-backed tensors.

3. 4acd37a59 - 2026-05-11 - Add uninitialized array allocation
   - Adds mx.empty(shape, dtype, stream=...).
   - 4 files, 60 insertions.
   - Chronologically this landed after the first DLPack commits, but it is the
     easiest independent PR and should be submitted first.

4. 3a6039d77 - 2026-05-13 - Add Python array bridge dylib
   - Adds mlx_python_bridge and an exported
     mlx_core_wrap_mx_array_move(mx::array*) symbol.
   - 2 files, 71 insertions.
   - This is useful for native extensions that already create C++ mx::array
     values and need to return Python mlx.core.array objects.
   - It is not required for the DLPack consumer and should be treated as a
     separate, optional design discussion.

### TileLang side

TileLang work that depends on the MLX PoC landed in this order:

- 2026-05-11 386a4552
  - broad Metal / MLX interop work, including MLX output allocation.
- 2026-05-11 9ee24a4b
  - use write-only MLX outputs for TVM-FFI, falling back to mx.zeros when
    mx.empty is unavailable.
- 2026-05-12 2ef8e64d
  - native MLX / TVM-FFI bridge for Metal kernel execution.
- 2026-05-13 d1d98e01, e11e1599, 93319ac6, 0de19234
  - graph-safe sync metadata, diagnostics, and device-event wiring.
- 2026-05-13 e23ca8fb
  - link the TileLang native MLX bridge against the new MLX Python array
    wrapper.

The TileLang tests exercising this path live mainly in:

- testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py

The core TileLang interop code is in:

- tilelang/contrib/mlx_interop.py
- tilelang/contrib/mlx_tvm_ffi.py
- tilelang/jit/adapter/tvm_ffi.py
- src/contrib/mlx_tvm_ffi/mlx_tvm_ffi_ext.cpp

---

## What the PoC Actually Implements

### 1. mx.empty

Files:

- mlx/ops.cpp
- mlx/ops.h
- python/src/ops.cpp
- python/tests/test_ops.py

Behavior:

- Adds mx.empty(shape, dtype=mx.float32, stream=None).
- Allocates an array without initializing the payload.
- Rejects negative dimensions.
- Keeps the existing MLX GPU float64 restriction.

Why it matters here:

- TileLang output buffers are write-only kernel results. Zero-filling them with
  mx.zeros is unnecessary work.
- tilelang.contrib.mlx_interop.mlx_metal_output(...) already uses
  mx.empty when present and falls back to mx.zeros for older MLX builds.

This can be upstreamed first because it is small and useful outside DLPack.

### 2. CPU DLPack Consumer

Files:

- python/src/dlpack_consumer.cpp
- python/src/dlpack_consumer.h
- python/src/dlpack_consumer_no_metal.cpp
- python/src/dlpack_format.h
- python/src/convert.cpp
- python/tests/test_dlpack_consumer.py

Behavior:

- Accepts either a raw PyCapsule or an object exposing __dlpack__.
- Recognizes both legacy dltensor and versioned dltensor_versioned
  capsules.
- Converts DLPack dtype to MLX dtype for scalar lanes.
- Requires row-contiguous layout.
- Rejects dtype override if it would require a copy or cast.
- Leaves rejected capsules unconsumed.
- Renames consumed capsules to the correct used-capsule name and calls the
  DLPack deleter when MLX releases the wrapping storage.
- Wraps CPU producer pointers through mx::allocator::make_buffer(...); if the
  active allocator cannot expose the pointer zero-copy, the import fails rather
  than silently staging a copy.

Known upstream-readiness gap:

- The code recognizes dltensor_versioned, but it does not yet validate
  DLPackVersion major/minor compatibility. Current DLPack headers define
  version 1.3 and explicitly require safe handling of major-version mismatch.
  A PR should add that check before merge.

### 3. Metal DLPack Consumer

Files:

- python/src/dlpack_consumer_metal.cpp
- mlx/backend/metal/custom_kernel.cpp
- mlx/backend/metal/device.cpp
- mlx/backend/metal/device.h
- python/src/array.cpp
- python/src/convert.cpp
- python/src/convert.h
- python/src/metal.cpp
- python/tests/test_array.py
- python/tests/test_device.py
- python/tests/test_fast.py
- python/tests/test_dlpack_consumer.py

Behavior:

- Accepts kDLMetal tensors.
- Treats DLTensor.data as an MTL::Buffer*.
- Requires MTLStorageModeShared.
- Rejects Managed and Private storage today.
- Rejects non-zero byte_offset.
- Rejects non-row-contiguous strides.
- Checks that shape and dtype fit inside the exported MTLBuffer.
- Wraps the foreign MTL::Buffer* directly in MLX storage and lets the DLPack
  owner lifetime keep the producer allocation alive.

Important scope note:

- The PoC does **not** currently accept MTLStorageModeManaged.
- The code accepts Shared only.

Why Shared-only is a reasonable first cut:

- MLX arrays normally use shared Metal buffers on Apple Silicon.
- TileLang / TVM-FFI can export shared-mode Metal buffers for MLX.
- Private buffers would need an explicit copy or command-buffer-mediated
  synchronization path; that is a separate design.

### 4. Python Array Bridge

Files:

- python/src/array_wrapper.cpp
- python/src/CMakeLists.txt

Behavior:

- Builds libmlx_python_bridge.dylib.
- Exposes mlx_core_wrap_mx_array_move(mx::array*).
- Lets a native extension create a C++ mx::array and return a Python
  mlx.core.array without going through DLPack.

This is useful for TileLang's native MLX graph primitive, but it is not part of
the minimal DLPack consumer story. It should be a later PR only if MLX
maintainers want a supported extension ABI for returning native arrays.

---

## What This Does Not Claim

- It does not make PyTorch MPS DLPack work. PyTorch still raises when asked to
  export MPS tensors via __dlpack__, as noted in #2848.
- It does not import kDLCUDA tensors. The PoC explicitly rejects CUDA DLPack.
- It does not implement arbitrary strided imports. Non-row-contiguous inputs
  are rejected with a clear error.
- It does not implement hidden dtype conversion. DLPack import is zero-copy or
  an error.
- It does not implement the new DLPack C exchange API
  (__dlpack_c_exchange_api__). That can be a future optimization.
- It does not yet solve GPU stream synchronization in a general cross-framework
  way. The current Metal path is sufficient for the TileLang / MLX flow, but
  upstream should decide the public contract.

---

## Promised vs Done

This section is deliberately blunt. It separates what the original downstream
story implied from what is actually finished today and what must be completed
before asking MLX maintainers to merge anything.

- mx.empty exists and avoids zero-fill for write-only outputs.
  - Status: done in 4acd37a59; TileLang uses it with a mx.zeros
    fallback.
  - Finish: rebase, keep small, add or keep shape and dtype tests.

- MLX can consume CPU DLPack producers.
  - Status: mostly done in c0cda6e20; raw capsules and producer objects
    work for row-contiguous CPU tensors.
  - Finish: add public mx.from_dlpack, add version checks, make error
    policy final.

- MLX can consume Metal DLPack producers zero-copy.
  - Status: partially done in 41ec3f5a8; works for kDLMetal, Shared
    storage, row-contiguous layout, and zero byte offset.
  - Finish: decide storage-mode policy, stream semantics, and whether byte
    offsets stay rejected.

- The implementation supports DLPack versioned capsules.
  - Status: partially done; recognizes dltensor_versioned.
  - Finish: validate DLPackVersion major/minor before reading fields beyond
    the safe prefix.

- The RFC answers #2848 for MPS/CUDA arrays.
  - Status: partially true. It answers the Metal/MPS-compatible producer case,
    not CUDA. PyTorch MPS remains blocked by PyTorch.
  - Finish: phrase #2848 as CPU + Metal progress; split CUDA into a future PR.

- mx.from_dlpack(obj) is available.
  - Status: not done. The PoC only has an internal C++ dlpack_to_mlx(...)
    and implicit mx.array(...) dispatch.
  - Finish: add explicit Python binding and tests before PR 2.

- mx.array(obj) automatically consumes DLPack producers.
  - Status: done in the PoC, but this is an API-policy choice.
  - Finish: keep behind PR 4 or fold into PR 2 only if maintainers want
    implicit dispatch.

- MTLStorageModeShared and Managed both work.
  - Status: not done. Current code accepts Shared only and rejects Managed /
    Private.
  - Finish: either update the code to support Managed safely or keep the RFC
    Shared-only.

- kDLCUDA works.
  - Status: not done. Current code explicitly rejects kDLCUDA.
  - Finish: separate CUDA design/PR for MLX CUDA builds.

- Non-contiguous DLPack tensors work.
  - Status: not done. Current code rejects non-row-contiguous strides.
  - Finish: either keep rejection as first-cut policy or implement stride-aware
    MLX storage/view semantics.

- Non-zero Metal byte_offset works.
  - Status: not done. Current Metal path rejects non-zero byte_offset.
  - Finish: decide whether MLX storage can represent imported offset safely;
    otherwise keep explicit rejection.

- Dtype conversion during import works.
  - Status: not done by design. The PoC rejects dtype overrides that require
    copy/cast.
  - Finish: keep zero-copy-only semantics or add explicit copy mode later.

- General cross-framework GPU stream synchronization is solved.
  - Status: not done. TileLang has graph/device-event handling for its path,
    but MLX DLPack import has no generic stream contract yet.
  - Finish: define __dlpack__(stream=...) semantics for Metal before broad
    device interop claims.

- Native extension can return Python mlx.core.array from C++ mx::array.
  - Status: done downstream in 3a6039d77, but it is separate from DLPack.
  - Finish: decide whether MLX wants this ABI; otherwise keep downstream-only.

The main unfinished items are therefore:

1. Add an explicit mx.from_dlpack API.
2. Add DLPack version compatibility validation.
3. Decide and document Metal stream semantics.
4. Decide whether Metal import remains Shared-only.
5. Keep CUDA, PyTorch MPS bypasses, arbitrary strides, dtype conversion, and
   native Python array wrapping out of the first consumer PR unless MLX
   maintainers explicitly ask for them.

---

## Completion Plan

### Before PR 1 (mx.empty)

- Rebase the mx.empty commit on current ml-explore/mlx@main.
- Keep the patch independent from DLPack.
- Verify CPU and Metal behavior for at least:
  - default dtype.
  - explicit dtype.
  - negative shape rejection.
  - GPU float64 rejection, matching existing MLX policy.

### Before PR 2 (CPU mx.from_dlpack)

- Add the public Python binding mx.from_dlpack(obj).
- Route it to the existing dlpack_to_mlx(...) implementation.
- Add explicit tests for:
  - raw legacy dltensor capsule.
  - producer object exposing __dlpack__.
  - consumed capsule rejection.
  - rejected capsule remains unconsumed.
  - dtype mismatch rejection.
  - non-row-contiguous rejection.
  - versioned capsule path.
- Add DLPack major/minor validation:
  - if major mismatches, call the deleter and fail without reading unsafe
    fields.
  - if minor is newer, allow only if MLX understands the fields it uses.
- Decide whether CPU import may fall back to a copy. The current PoC is
  zero-copy-or-error.

### Before PR 3 (kDLMetal)

- Keep the first Metal PR Shared-only unless maintainers request Managed.
- Add tests for:
  - Shared storage import.
  - Private / Managed rejection, or supported behavior if policy changes.
  - non-zero byte offset rejection.
  - non-row-contiguous rejection.
  - shape/dtype requiring more bytes than buffer length.
  - non-Metal build rejecting kDLMetal cleanly.
- Define how __dlpack__(stream=...) should map to MLX Metal command-buffer
  behavior, or explicitly state that first-cut import assumes producer-side
  synchronization.

### Before PR 4 (mx.array implicit dispatch)

- Decide protocol precedence with maintainers.
- Add tests for objects exposing multiple protocols:
  - __mlx_array__ plus __dlpack__.
  - NumPy array protocol plus __dlpack__.
  - raw PyCapsule.
  - plain DLPack producer object.
- Ensure mx.array(obj, dtype=...) has a clear copy/cast policy for DLPack
  inputs.

### Before PR 5 (native Python array bridge)

- Decide whether MLX wants to expose a native extension ABI at all.
- If yes, document ownership:
  - caller passes new mx::array(...).
  - mlx_core_wrap_mx_array_move takes ownership.
  - returned Python object owns the moved C++ array.
- Add a minimal external-extension style test, not only in-tree use.

---

## Proposed Upstream PR Sequence

### PR 1: [ops] Add mx.empty

Scope:

- mx.empty(shape, dtype=..., stream=...)
- Python binding and tests.

Why first:

- Small diff.
- Useful independently.
- Lets downstream code allocate write-only output buffers without a zero-fill.

### PR 2: [Python] Add explicit DLPack consumer for CPU

Recommended API:

- Add mx.from_dlpack(obj) first.
- Optionally wire mx.array(obj) to call the same path after maintainers agree
  on implicit dispatch precedence.

Why explicit first:

- Matches NumPy and JAX.
- Avoids surprising mx.array(...) behavior for objects that expose multiple
  protocols.
- Gives tests a clear target for ownership and error-path behavior.

Scope:

- DLPack capsule / producer parsing.
- CPU zero-copy import.
- consumed-capsule lifetime handling.
- dtype/shape/stride validation.
- versioned capsule compatibility checks.
- no Metal, no CUDA.

### PR 3: [Metal] Add kDLMetal DLPack consumer

Scope:

- kDLMetal import behind MLX_BUILD_METAL.
- MTL::Buffer* wrapping.
- Shared storage mode only for the first PR.
- clear errors for Private, Managed, non-zero byte offset, and
  non-row-contiguous strides.
- tests skipped when Metal is unavailable.

Open design point:

- Whether Managed should be accepted on Intel-era Macs or rejected until
  explicit coherency handling exists.
- Whether Private should fail or stage through a copy.
- What __dlpack__(stream=...) should mean for Metal command buffers.

### PR 4: mx.array(...) implicit dispatch

Scope:

- Make mx.array(obj) consume DLPack producers when appropriate.
- Preserve MLX's existing protocol precedence.

Current PoC precedence:

1. native scalars/lists/tuples.
2. existing mlx.core.array.
3. raw DLPack capsule.
4. DLPack producer if no __mlx_array__ or NumPy array protocol is present.
5. nanobind ndarray / NumPy array path.
6. __mlx_array__.
7. final DLPack fallback.
8. generic accessor path.

That precedence works for our tests, but maintainers should choose the final
policy.

### PR 5: optional native Python array bridge

Scope:

- libmlx_python_bridge.dylib.
- mlx_core_wrap_mx_array_move(mx::array*).

This is not required for DLPack. It is useful for native extensions, including
TileLang's MLX graph primitive, but it creates a public-ish ABI surface. It
should be discussed separately.

### Future PR: CUDA DLPack Consumer

If MLX wants the CUDA half of #2848 addressed:

- add kDLCUDA support for Linux CUDA builds.
- map __dlpack__(stream=...) to MLX CUDA stream semantics.
- decide whether mx.array(cupy_array) should zero-copy import CuPy CUDA
  buffers or require explicit mx.from_dlpack.

The current PoC intentionally does not do this.

---

## Open Questions for MLX Maintainers

1. Should the public API be explicit only (mx.from_dlpack), implicit only
   (mx.array), or both?
2. Should mx.array(...) auto-dispatch to DLPack only after __mlx_array__
   and NumPy-style protocols, as in the PoC?
3. For kDLMetal, is MTLStorageModeShared the only storage mode MLX wants to
   accept initially?
4. Should Managed ever be accepted, and if yes what coherency contract should
   the producer satisfy?
5. Should Private fail clearly, or should MLX stage through an explicit copy?
6. What should the Metal interpretation of DLPack's stream argument be?
7. What DLPack version floor should MLX enforce? Current DLPack headers are
   1.3; the PoC needs explicit version validation before merge.
8. Should MLX core expose a native extension ABI for wrapping C++ mx::array
   into Python mlx.core.array, or should that stay downstream-only?
9. Does MLX want CUDA DLPack consumer support in the same RFC, or should CUDA
   be a separate issue after CPU + Metal land?

---

## What We Are Offering

If the maintainers are interested, we can:

- open PR 1 (mx.empty) immediately.
- rebase PR 2 on current ml-explore/mlx@main and add explicit
  mx.from_dlpack.
- tighten PR 2 with DLPack major/minor validation before review.
- split Metal into PR 3 with Shared-only semantics.
- keep the Python array bridge out of the initial DLPack PRs unless the team
  wants that native-extension hook.
- keep CUDA out of the initial PRs unless the team wants to broaden #2848 now.

If the MLX team prefers to implement the feature themselves, the downstream
code is public and we can walk through the design and test cases.

---

## References

- MLX issue #2848:
  https://github.com/ml-explore/mlx/issues/2848
- DLPack header used by the TileLang / TVM-FFI tree:
  3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack/dlpack.h
- Downstream MLX PoC:
  https://github.com/DatasunriseOU/mlx
- TileLang Metal / MLX tests:
  testing/python/metal/test_tvm_ffi_metal_stream_dlpack.py
