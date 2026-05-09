/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton {

/// Common interface for tensor descriptor types.
/// 
///     This interface provides a unified API for different tensor descriptor
///     implementations (e.g., tiled TensorDescType, im2col TensorDescIm2ColType).
///     All tensor descriptors share the concept of a "block type" which describes
///     the shape and element type of the data block being accessed.
/// 
///     Concrete implementations:
///     - TensorDescType (Triton dialect): Basic tiled tensor descriptor
///     - TensorDescIm2ColType (TritonNvidiaGPU dialect): Im2col tensor descriptor
///       with additional convolution parameters
class TensorDescInterface;

} // namespace mlir::triton
namespace mlir::triton {

/// Common interface for tensor descriptor types.
/// 
///     This interface provides a unified API for different tensor descriptor
///     implementations (e.g., tiled TensorDescType, im2col TensorDescIm2ColType).
///     All tensor descriptors share the concept of a "block type" which describes
///     the shape and element type of the data block being accessed.
/// 
///     Concrete implementations:
///     - TensorDescType (Triton dialect): Basic tiled tensor descriptor
///     - TensorDescIm2ColType (TritonNvidiaGPU dialect): Im2col tensor descriptor
///       with additional convolution parameters
namespace detail {
struct TensorDescInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    mlir::RankedTensorType (*getBlockType)(const Concept *impl, ::mlir::Type );
    mlir::RankedTensorType (*getSignlessBlockType)(const Concept *impl, ::mlir::Type );
  };
  template<typename ConcreteType>
  class Model : public Concept {
  public:
    using Interface = ::mlir::triton::TensorDescInterface;
    Model() : Concept{getBlockType, getSignlessBlockType} {}

    static inline mlir::RankedTensorType getBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::RankedTensorType getSignlessBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteType>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::triton::TensorDescInterface;
    FallbackModel() : Concept{getBlockType, getSignlessBlockType} {}

    static inline mlir::RankedTensorType getBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
    static inline mlir::RankedTensorType getSignlessBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteType;
    mlir::RankedTensorType getSignlessBlockType(::mlir::Type tablegen_opaque_val) const;
  };
};
template <typename ConcreteType>
struct TensorDescInterfaceTrait;

} // namespace detail
class TensorDescInterface : public ::mlir::TypeInterface<TensorDescInterface, detail::TensorDescInterfaceInterfaceTraits> {
public:
  using ::mlir::TypeInterface<TensorDescInterface, detail::TensorDescInterfaceInterfaceTraits>::TypeInterface;
  template <typename ConcreteType>
  struct Trait : public detail::TensorDescInterfaceTrait<ConcreteType> {};

  /// Returns the block type of the tensor descriptor
  mlir::RankedTensorType getBlockType() const;

  /// Returns the block type with signless integer element type
  mlir::RankedTensorType getSignlessBlockType() const;
};

} // namespace mlir::triton
namespace mlir::triton::detail {

  template <typename ConcreteType>
  struct TensorDescInterfaceTrait : public ::mlir::TypeInterface<TensorDescInterface, detail::TensorDescInterfaceInterfaceTraits>::Trait<ConcreteType> {

    /// Returns the block type with signless integer element type
    mlir::RankedTensorType getSignlessBlockType() const {
      auto resTy = (*static_cast<const ConcreteType *>(this)).getBlockType();
        if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(resTy.getElementType())) {
          auto width = resTy.getElementTypeBitWidth();
          auto signlessTy = mlir::IntegerType::get((*static_cast<const ConcreteType *>(this)).getContext(), width);
          resTy = resTy.clone(signlessTy);
        }
        return resTy;
    }
  };

} // namespace mlir::triton::detail
namespace mlir::triton {

template<typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::Model<ConcreteType>::getBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getBlockType();
}
template<typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::Model<ConcreteType>::getSignlessBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getSignlessBlockType();
}
template<typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getBlockType(tablegen_opaque_val);
}
template<typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::FallbackModel<ConcreteType>::getSignlessBlockType(const Concept *impl, ::mlir::Type tablegen_opaque_val) {
  return static_cast<const ConcreteType *>(impl)->getSignlessBlockType(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteType>
mlir::RankedTensorType detail::TensorDescInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteType>::getSignlessBlockType(::mlir::Type tablegen_opaque_val) const {
auto resTy = (::llvm::cast<ConcreteType>(tablegen_opaque_val)).getBlockType();
        if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(resTy.getElementType())) {
          auto width = resTy.getElementTypeBitWidth();
          auto signlessTy = mlir::IntegerType::get((::llvm::cast<ConcreteType>(tablegen_opaque_val)).getContext(), width);
          resTy = resTy.clone(signlessTy);
        }
        return resTy;
}

} // namespace mlir::triton
