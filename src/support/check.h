#ifndef TILELANG_SUPPORT_CHECK_H_
#define TILELANG_SUPPORT_CHECK_H_

#include <tvm/ffi/tvm_ffi.h>

// Fork integration (Task 11 / #2216): upstream's `support/check.h` introduced
// 3-arg CHECK_*(x, y, ErrorKind) macros. The fork keeps the legacy 2-arg form
// because `src/transform/vendored/tl_compat.h` is force-included project-wide
// and already provides `CHECK_*(x, y)` -> `TVM_FFI_*(x, y, InternalError)`.
// Only define each macro here if tl_compat (or whoever else) has not already
// installed a definition, so the existing call sites keep compiling.
#ifndef CHECK
#define CHECK(cond, ErrorKind) TVM_FFI_CHECK(cond, ErrorKind)
#endif
#ifndef CHECK_LT
#define CHECK_LT(x, y, ErrorKind) TVM_FFI_CHECK_LT(x, y, ErrorKind)
#endif
#ifndef CHECK_GT
#define CHECK_GT(x, y, ErrorKind) TVM_FFI_CHECK_GT(x, y, ErrorKind)
#endif
#ifndef CHECK_LE
#define CHECK_LE(x, y, ErrorKind) TVM_FFI_CHECK_LE(x, y, ErrorKind)
#endif
#ifndef CHECK_GE
#define CHECK_GE(x, y, ErrorKind) TVM_FFI_CHECK_GE(x, y, ErrorKind)
#endif
#ifndef CHECK_EQ
#define CHECK_EQ(x, y, ErrorKind) TVM_FFI_CHECK_EQ(x, y, ErrorKind)
#endif
#ifndef CHECK_NE
#define CHECK_NE(x, y, ErrorKind) TVM_FFI_CHECK_NE(x, y, ErrorKind)
#endif
#ifndef CHECK_NOTNULL
#define CHECK_NOTNULL(x, ErrorKind) TVM_FFI_CHECK_NOTNULL(x, ErrorKind)
#endif

#ifndef ICHECK
#define ICHECK(x) TVM_FFI_ICHECK(x)
#endif
#ifndef ICHECK_LT
#define ICHECK_LT(x, y) TVM_FFI_ICHECK_LT(x, y)
#endif
#ifndef ICHECK_GT
#define ICHECK_GT(x, y) TVM_FFI_ICHECK_GT(x, y)
#endif
#ifndef ICHECK_LE
#define ICHECK_LE(x, y) TVM_FFI_ICHECK_LE(x, y)
#endif
#ifndef ICHECK_GE
#define ICHECK_GE(x, y) TVM_FFI_ICHECK_GE(x, y)
#endif
#ifndef ICHECK_EQ
#define ICHECK_EQ(x, y) TVM_FFI_ICHECK_EQ(x, y)
#endif
#ifndef ICHECK_NE
#define ICHECK_NE(x, y) TVM_FFI_ICHECK_NE(x, y)
#endif
#ifndef ICHECK_NOTNULL
#define ICHECK_NOTNULL(x) TVM_FFI_ICHECK_NOTNULL(x)
#endif

#ifndef DCHECK
#define DCHECK(x) TVM_FFI_DCHECK(x)
#endif
#ifndef DCHECK_LT
#define DCHECK_LT(x, y) TVM_FFI_DCHECK_LT(x, y)
#endif
#ifndef DCHECK_GT
#define DCHECK_GT(x, y) TVM_FFI_DCHECK_GT(x, y)
#endif
#ifndef DCHECK_LE
#define DCHECK_LE(x, y) TVM_FFI_DCHECK_LE(x, y)
#endif
#ifndef DCHECK_GE
#define DCHECK_GE(x, y) TVM_FFI_DCHECK_GE(x, y)
#endif
#ifndef DCHECK_EQ
#define DCHECK_EQ(x, y) TVM_FFI_DCHECK_EQ(x, y)
#endif
#ifndef DCHECK_NE
#define DCHECK_NE(x, y) TVM_FFI_DCHECK_NE(x, y)
#endif
#ifndef DCHECK_NOTNULL
#define DCHECK_NOTNULL(x) TVM_FFI_DCHECK_NOTNULL(x)
#endif

#endif // TILELANG_SUPPORT_CHECK_H_
