#include "codegen_cuda.h"
// apache/tvm-latest stripped the public `runtime/cuda/cuda_module.h`; the
// codegen-facing factory now lives in `target/cuda/cuda_fallback_module.h`
// (`target::CUDAModuleCreateWithFallback`).  It returns a real `CUDAModuleNode`
// when the runtime is registered ("ffi.Module.create.cuda") and a
// `CUDAFallbackModuleNode` otherwise — equivalent surface to the old
// `runtime::CUDAModuleCreate`.
#include "target/cuda/cuda_fallback_module.h"
// apache/tvm-latest renamed `runtime/meta_data.h` → `runtime/metadata.h`.
#include "runtime/metadata.h"
#include "runtime/pack_args.h"
#include "transform/common/attr.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace codegen {

static std::string GetDeviceGlobalSymbol(const GlobalVar &gvar,
                                         const tirx::PrimFunc &f) {
  if (auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
    return static_cast<std::string>(global_symbol.value());
  }
  return gvar->name_hint;
}

static void ValidateUniqueDeviceGlobalSymbols(const IRModule &mod) {
  std::unordered_map<std::string, std::string> symbol_to_gvar;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tirx::PrimFunc>(kv.second);
    std::string global_symbol = GetDeviceGlobalSymbol(gvar, f);

    auto [it, inserted] =
        symbol_to_gvar.emplace(global_symbol, gvar->name_hint);
    ICHECK(inserted)
        << "Duplicate CUDA kernel global_symbol `" << global_symbol
        << "` found on PrimFuncs `" << it->second << "` and `"
        << gvar->name_hint
        << "`. T.CUDASourceCodeKernel emits raw CUDA source without "
           "renaming, so CUDA entry names must be unique within the compiled "
           "module.";
  }
}

// apache/tvm-latest converted `runtime::FunctionInfo` from a struct with
// std::vector members to an ObjectRef around `FunctionInfoObj` whose fields
// are `ffi::Array<...>`.  Build up the ffi::Array values, then construct the
// FunctionInfo via its 4-arg ctor (name, arg_types, launch_param_tags,
// arg_extra_tags).
static ffi::Map<ffi::String, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule &mod) {
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tirx::PrimFunc>(kv.second);

    // z3-final: validate params before constructing FunctionInfo.  An empty
    // params array is permitted (kernels can take zero args); but a defined
    // PrimFunc must not have null param Vars in the array.
    ICHECK(f.defined()) << "ExtractFuncInfo: PrimFunc is undefined";
    for (size_t i = 0; i < f->params.size(); ++i) {
      ICHECK(f->params[i].defined())
          << "ExtractFuncInfo: PrimFunc has undefined param at index " << i;
    }

    ffi::Array<DLDataType> arg_types;
    ffi::Array<runtime::ArgExtraTags> arg_extra_tags;
    auto is_tensormap = [](const tirx::Var &var) -> bool {
      const auto *type = var->type_annotation.as<PointerTypeNode>();
      if (type == nullptr) return false;
      return type->element_type.as<TensorMapTypeNode>() != nullptr;
    };
    for (size_t i = 0; i < f->params.size(); ++i) {
      // apache/tvm-latest dropped the `kDLGridConstant` synthetic dtype that
      // was previously emitted for `grid_constant` storage_scope params; the
      // tensor-map case is now signalled per-arg via
      // `ArgExtraTags::kTensorMap`.
      DataType dtype = f->params[i].dtype();
      // Device runtime cannot directly take bool arguments, map to int32.
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      arg_types.push_back(dtype);
      arg_extra_tags.push_back(is_tensormap(f->params[i])
                                   ? runtime::ArgExtraTags::kTensorMap
                                   : runtime::ArgExtraTags::kNone);
    }
    ffi::Array<ffi::String> launch_param_tags;
    if (f->HasNonzeroAttr(tl::attr::kHasGridSync)) {
      launch_param_tags.push_back(
          runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      launch_param_tags.push_back(
          runtime::launch_param::kUseCooperativeLaunch);
    }
    if (f->GetAttr<ffi::Array<Integer>>("cluster_dims").defined()) {
      launch_param_tags.push_back(runtime::launch_param::kClusterDimX);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimY);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimZ);
    }
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        if (tag != runtime::launch_param::kClusterDimX &&
            tag != runtime::launch_param::kClusterDimY &&
            tag != runtime::launch_param::kClusterDimZ) {
          launch_param_tags.push_back(tag);
        }
      }
    }
    ffi::String global_symbol =
        GetDeviceGlobalSymbol(Downcast<GlobalVar>(kv.first), f);
    fmap.Set(global_symbol,
             runtime::FunctionInfo(global_symbol, std::move(arg_types),
                                   std::move(launch_param_tags),
                                   std::move(arg_extra_tags)));
  }
  return fmap;
}

ffi::Module BuildTileLangCUDA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_validate")) {
    (*f)(mod);
  }

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCUDA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_compile")) {
    // Fetch current pass context config and pass into the compile callback
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    ptx = (*f)(code, target, pass_ctx->config).cast<std::string>();
    if (ptx[0] != '/')
      fmt = "cubin";
  } else {
    ICHECK(0);
  }
  // Hand off compiled bytes to the fallback-aware factory (apache/tvm-latest
  // dropped the public `runtime::CUDAModuleCreate` wrapper).
  ffi::Map<ffi::String, ffi::String> source_map;
  source_map.Set("cuda", code);
  source_map.Set("cuda_source", code);
  return target::CUDAModuleCreateWithFallback(ffi::Bytes(ptx.data(), ptx.size()),
                                              ffi::String(fmt), ExtractFuncInfo(mod),
                                              source_map);
}

ffi::Module BuildTileLangCUDAWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_validate")) {
    (*f)(mod);
  }

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCUDA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  ffi::Map<ffi::String, ffi::String> source_map;
  source_map.Set("cuda", code);
  source_map.Set("cuda_source", code);
  return target::CUDAModuleCreateWithFallback(ffi::Bytes("ptx", 3), ffi::String("ptx"),
                                              ExtractFuncInfo(mod), source_map);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_cuda", BuildTileLangCUDA)
      .def("target.build.tilelang_cuda_without_compile",
           BuildTileLangCUDAWithoutCompile);
}

} // namespace codegen
} // namespace tvm
