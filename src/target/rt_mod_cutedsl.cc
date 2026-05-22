#include "codegen_cutedsl.h"
#include "target/cuda/cuda_fallback_module.h"
#include "runtime/pack_args.h"
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace codegen {

static ffi::Map<ffi::String, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule &mod) {
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tirx::PrimFunc>(kv.second);

    ffi::Array<DLDataType> arg_types;
    ffi::Array<runtime::ArgExtraTags> arg_extra_tags;
    auto is_tensormap = [](const tirx::Var &var) -> bool {
      const auto *type = var->type_annotation.as<PointerTypeNode>();
      if (type == nullptr) return false;
      return type->element_type.as<TensorMapTypeNode>() != nullptr;
    };
    for (size_t i = 0; i < f->params.size(); ++i) {
      DataType dtype = f->params[i].dtype();
      if (dtype.is_bool()) dtype = DataType::Int(32);
      arg_types.push_back(dtype);
      arg_extra_tags.push_back(is_tensormap(f->params[i])
                                   ? runtime::ArgExtraTags::kTensorMap
                                   : runtime::ArgExtraTags::kNone);
    }
    ffi::Array<ffi::String> launch_param_tags;
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        launch_param_tags.push_back(tag);
      }
    }
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    ffi::String name = global_symbol.value();
    fmap.Set(name,
             runtime::FunctionInfo(name, std::move(arg_types),
                                   std::move(launch_param_tags),
                                   std::move(arg_extra_tags)));
  }
  return fmap;
}

ffi::Module BuildTileLangCuTeDSLWithoutCompile(IRModule mod, Target target) {
  CodeGenTileLangCuTeDSL cg;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCuTeDSL: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cutedsl_postproc")) {
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
  refl::GlobalDef().def("target.build.tilelang_cutedsl_without_compile",
                        BuildTileLangCuTeDSLWithoutCompile);
}

} // namespace codegen
} // namespace tvm
