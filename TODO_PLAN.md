# TileLang Migration & Unified Fused-Kernel Plan (40 Steps)

## Group 1: TileLang Compiler & CodeGen (Архитектурные изменения)
- [x] Task 1: Поддержка динамических границ цикла в T.Parallel (valid_block_H) для Sparse GQA. Исключить необходимость внутреннего if.
- [x] Task 2: Multi-buffering для TMA (DeepSeek MLA/NSA). Двойная/тройная буферизация для bar_q, bar_k барьеров.
- [x] Task 3: Layout Inference для INT8 (Dequantize GEMM). Улучшение анализа коалесцирования памяти для 4D-загрузок int8.
- [x] Task 4: Warp Specialization (Flash Decoding). Изоляция TMA-варпов от Math-варпов без кросс-варп зависимостей.

## Group 2: Triton Frontend (Python & C++)
- [x] Task 5: Структурный парсинг сигнатур (map_tt_func). Генератор TIR-сигнатур из аргументов функции TTIR вместо неявных эпилогов.
- [x] Task 6: Интеграция C++ MLIR (Integration #5). Полноценный вендоринг апстрим-диалектов Triton, удаление #ifdef в CMakeLists.txt и ptr_analysis_shim.cc.
- [x] Task 7: PtrAnalysis в эмиттерах памяти. Перевод map_tt_load/store в op_mapping.py на результаты PtrAnalysis.
- [x] Task 8: Реализация tt.split и tt.join в маппере Triton -> TileLang.
- [x] Task 9: Реализация tt.histogram в маппере Triton -> TileLang.
- [x] Task 10: Реализация tt.print с %n-санитизацией в маппере Triton -> TileLang.
- [x] Task 11: Поддержка async-copy (cp.async) / barrier в маппере Triton.
- [x] Task 12: Пакет conformance-тестов Triton-frontend: vector_add, softmax, matmul.
- [x] Task 13: Пакет conformance-тестов Triton-frontend: layer_norm (Welford).
- [x] Task 14: Пакет conformance-тестов Triton-frontend: Flash Attention v2.

## Group 3: Z3 Prover & Transform Fixes
- [x] Task 15: Устранение регрессии Z3 Prover для CUDA / gb10 (исследование TILELANG_DISABLE_Z3).
- [x] Task 16: Фикс Z3 в Loop Vectorize (выравнивание и unit-stride доказательства).
- [x] Task 17: Фикс Z3 в Drop Provable Bound Checks (BV32 fallback).
- [x] Task 18: Исправление mul-kind handling в C++ пассе reduce_prod.

## Group 4: Kernel Implementations & Optimizations
- [x] Task 19: FP8 amax port: инъекция `__globals__` для решения проблемы с `get_type_hints` closure-rebind.
- [x] Task 20: dsa_splitk: реализация tiled Q-cache для продакшен AH-формфакторов. (Skipped: external repository cppmega.mlx)
- [x] Task 21: Интеграция ATEN_DISPATCH для Flash Attention CPU (`aten._scaled_dot_product_flash_attention_for_cpu`).
- [x] Task 22: Расширение `tt.dot trans_b` тестами на вывод лейаутов.

## Group 5: torch.compile Backend
- [x] Task 23: Создание скелета интеграции `torch.compile(backend="tilelang")`.
- [x] Task 24: Маппинг узлов FX -> TileLang op для matmul и layernorm.
- [x] Task 25: Маппинг узлов FX -> TileLang op для softmax и gelu.
- [x] Task 26: Маппинг узлов FX -> TileLang op для примитивов attention.
- [x] Task 27: Обертка `torch.library.custom_op` для autograd meta.
- [x] Task 28: AOT autograd `register_double_backward` с аналитическим zero-grad аккумулятором.
- [x] Task 29: Кэш с группировкой по формам (shape-bucketed cache) для autotune shortlist.
- [x] Task 30: Внедрение потокобезопасного `_AUTOTUNE_LOCK` для AOT autograd.

## Group 6: Extern Intrinsic Mechanism
- [x] Task 31: Реализация декоратора `tl.extern_intrinsic` и TIR builder.
- [x] Task 32: Диспетчеризация тел ядер (per-target body dispatch) в генераторе кода.
- [x] Task 33: Создание референсного ядра CUDA для тестирования cross-fusion.
- [x] Task 34: Создание референсного ядра HIP для тестирования cross-fusion.
- [x] Task 35: Создание референсного ядра Metal для тестирования cross-fusion.

## Group 7: TMA & Cross-Platform Memory
- [x] Task 36: Декомпозиция TMA fallback в арифметику указателей для Apple Metal (non-NV).
- [x] Task 37: Декомпозиция TMA fallback в арифметику указателей для AMD HIP (non-NV).
- [x] Task 38: Кэширование на уровне Python-биндингов PtrAnalysis.
- [x] Task 39: Дедупликация rewrite-error в PtrAnalysis.
- [x] Task 40: Поддержка обхода `tts.make_gather_scatter_tptr` в PtrAnalysis.
