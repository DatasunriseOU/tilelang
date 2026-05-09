/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TritonNvidiaGPUAttrDefs.td                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::triton::nvidia_gpu {

// allowed 32-bit signless integer cases: 1, 2
enum class TMEMLoadReduceModifier : uint32_t {
  MIN = 1,
  MAX = 2,
};

::std::optional<TMEMLoadReduceModifier> symbolizeTMEMLoadReduceModifier(uint32_t);
::llvm::StringRef stringifyTMEMLoadReduceModifier(TMEMLoadReduceModifier);
::std::optional<TMEMLoadReduceModifier> symbolizeTMEMLoadReduceModifier(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForTMEMLoadReduceModifier() {
  return 2;
}


inline ::llvm::StringRef stringifyEnum(TMEMLoadReduceModifier enumValue) {
  return stringifyTMEMLoadReduceModifier(enumValue);
}

template <typename EnumType>
::std::optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::std::optional<TMEMLoadReduceModifier> symbolizeEnum<TMEMLoadReduceModifier>(::llvm::StringRef str) {
  return symbolizeTMEMLoadReduceModifier(str);
}

} // namespace mlir::triton::nvidia_gpu

namespace mlir {
template <typename T, typename>
struct FieldParser;

template<>
struct FieldParser<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier, ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> {
  template <typename ParserT>
  static FailureOr<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return parser.emitError(loc, "expected keyword for allowed 32-bit signless integer cases: 1, 2");

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>(enumKeyword))
      return *attr;
    return parser.emitError(loc, "expected one of [min, max] for allowed 32-bit signless integer cases: 1, 2, got: ") << enumKeyword;
  }
};

/// Support for std::optional, useful in attribute/type definition where the enum is
/// used as:
///
///    let parameters = (ins OptionalParameter<"std::optional<TheEnumName>">:$value);
template<>
struct FieldParser<std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>, std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>> {
  template <typename ParserT>
  static FailureOr<std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>> parse(ParserT &parser) {
    // Parse the keyword/string containing the enum.
    std::string enumKeyword;
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalKeywordOrString(&enumKeyword)))
      return std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>{};

    // Symbolize the keyword.
    if (::std::optional<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> attr = ::mlir::triton::nvidia_gpu::symbolizeEnum<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>(enumKeyword))
      return attr;
    return parser.emitError(loc, "expected one of [min, max] for allowed 32-bit signless integer cases: 1, 2, got: ") << enumKeyword;
  }
};
} // namespace mlir

namespace llvm {
inline ::llvm::raw_ostream &operator<<(::llvm::raw_ostream &p, ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier value) {
  auto valueStr = stringifyEnum(value);
  return p << valueStr;
}
} // namespace llvm

namespace llvm {
template<> struct DenseMapInfo<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier getEmptyKey() {
    return static_cast<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier getTombstoneKey() {
    return static_cast<::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier &lhs, const ::mlir::triton::nvidia_gpu::TMEMLoadReduceModifier &rhs) {
    return lhs == rhs;
  }
};
}

