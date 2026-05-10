import sys
import pytest
from poc.triton_frontend.ptr_analysis import PtrAnalysis, dialects_available

DEDUP_MLIR = """
module {
  tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: i32) {
    %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %1 = tt.splat %arg1 : i32 -> tensor<16xi32>
    // This dynamic addptr cannot be resolved to a uniform base
    %2 = tt.addptr %0, %1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<f32>>
    
    // Duplicate it
    %4 = tt.addptr %0, %1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %5 = tt.load %4 : tensor<16x!tt.ptr<f32>>
    
    // And again
    %6 = tt.addptr %0, %1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %7 = tt.load %6 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
"""

@pytest.mark.skipif(
    not dialects_available(),
    reason="shim built without TritonStructured/Triton dialects",
)
def test_ptr_analysis_dedup_warnings(capsys):
    # This should trigger rewrite warnings, but they should be deduplicated
    pa = PtrAnalysis(DEDUP_MLIR)
    try:
        pa.rewrite()
    except Exception:
        pass
    
    # We can check that stderr doesn't contain the same line 3 times
    captured = capsys.readouterr()
    stderr = captured.err
    lines = stderr.strip().split('\n')
    
    # Count occurrences of any non-empty line
    counts = {}
    for line in lines:
        if line.strip():
            counts[line.strip()] = counts.get(line.strip(), 0) + 1
            
    # Under deduplication, no line should appear more than once (or twice if it's a generic prefix, 
    # but the actual diagnostic message should be deduped because we dedup the entire diagnostic string)
    for line, count in counts.items():
        if "PtrAnalysis" in line:
            assert count == 1, f"Line '{line}' appeared {count} times, expected 1"
