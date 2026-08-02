// CPPMEGA: this header used to expose a conservative `Z3ProverStub` that
// always returned -1 / false, forcing TileLang's sync rewriter to fall
// back to safe full-sync paths. It has been replaced by a real
// Z3-backed prover (see vendored/z3_prover.h). This file now exists
// purely so existing `#include "vendored/z3_prover_stub.h"` lines keep
// resolving — leave it in place to avoid touching call-site includes.
#ifndef TILELANG_VENDORED_Z3_PROVER_STUB_H_
#define TILELANG_VENDORED_Z3_PROVER_STUB_H_

#include "z3_prover.h"

#endif // TILELANG_VENDORED_Z3_PROVER_STUB_H_
