// =====================================================================
// PLACEHOLDER --- this file is NOT the real nlohmann/json.hpp.
//
// To enable the optional nlohmann::json encoder (CMake option
// TRITON_FRONTEND_USE_NLOHMANN_JSON=ON), vendor the upstream single-include
// header here, replacing this stub:
//
//   curl -L -o poc/triton_frontend/_cxx/third_party/nlohmann/json.hpp \
//        https://github.com/nlohmann/json/releases/latest/download/json.hpp
//
// The single header is MIT-licensed (Niels Lohmann). Pin the upstream tag
// you fetched in third_party/nlohmann/VERSION when you vendor it so the
// re-fetch is reproducible.
//
// Why a placeholder rather than committing the ~1MB header up front? The
// default build (option=OFF) doesn't need it; vendoring a megabyte-class
// dependency for an opt-in code path bloats the repo for everyone. The
// CMake option produces a hard FATAL_ERROR with the curl line above when
// the user forgets this step, so the failure mode is loud and self-fixing.
// =====================================================================

#ifndef TL_PA_NLOHMANN_JSON_PLACEHOLDER
#define TL_PA_NLOHMANN_JSON_PLACEHOLDER

#error \
"nlohmann/json.hpp placeholder included. Vendor the real single-include " \
"header before building with -DTRITON_FRONTEND_USE_NLOHMANN_JSON=ON. " \
"See poc/triton_frontend/_cxx/third_party/nlohmann/json.hpp for instructions."

#endif  // TL_PA_NLOHMANN_JSON_PLACEHOLDER
