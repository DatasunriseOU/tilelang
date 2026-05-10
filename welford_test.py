import triton
import triton.language as tl

@triton.jit
def _welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    # w2_over_w = weight_2 / new_weight  # triton might not support multiple return?
    return (
        mean_1 + delta * (weight_2 / new_weight),
        m2_1 + m2_2 + delta * delta * weight_1 * (weight_2 / new_weight),
        new_weight,
    )

print("Parsed")
