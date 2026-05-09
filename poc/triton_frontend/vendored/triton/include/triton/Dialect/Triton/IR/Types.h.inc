/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::triton {

/// Pointer type (`::mlir::triton::PointerType`) in Triton IR type system
/// Pointer type in Triton IR type system. Pointer types may only point to scalar element types.
class PointerType;
/// Tensor descriptor type (`::mlir::triton::TensorDescType`) in Triton IR type system
/// A portable abstraction for TMA descriptors.
///       This is the base tensor descriptor type for tiled tensor memory access.
/// 
///       For specialized access patterns like im2col, see TensorDescIm2ColType
///       in the TritonNvidiaGPU dialect.
class TensorDescType;
namespace detail {

struct PointerTypeStorage;

} // namespace detail
class PointerType : public ::mlir::Type::TypeBase<PointerType, ::mlir::Type, detail::PointerTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tt.ptr";
  static constexpr ::llvm::StringLiteral dialectName = "tt";
  using Base::getChecked;
  static PointerType get(Type pointeeType, int addressSpace);
  static PointerType getChecked(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type pointeeType, int addressSpace);
  static ::llvm::LogicalResult verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type pointeeType, int addressSpace);
  static ::llvm::LogicalResult verifyInvariants(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, Type pointeeType, int addressSpace);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ptr"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Type getPointeeType() const;
  int getAddressSpace() const;
};
namespace detail {

struct TensorDescTypeStorage;

} // namespace detail
class TensorDescType : public ::mlir::Type::TypeBase<TensorDescType, ::mlir::Type, detail::TensorDescTypeStorage, ::mlir::triton::TensorDescInterface::Trait> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "tt.tensordesc";
  static constexpr ::llvm::StringLiteral dialectName = "tt";
  static TensorDescType get(::mlir::MLIRContext *context, RankedTensorType blockType);
  static TensorDescType get(::mlir::MLIRContext *context, RankedTensorType blockType, bool isSigned);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensordesc"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  RankedTensorType getBlockType() const;
};

} // namespace mlir::triton
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::PointerType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::triton::TensorDescType)

#endif // GET_TYPEDEF_CLASSES

