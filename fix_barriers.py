import re

def process_file(filename):
    with open(filename, "r") as f:
        content = f.read()

    # 1. Allocations
    alloc_pattern = r'''            KV_shared_0_l = T\.alloc_shared\(\[(.+?),\s*(.+?)\], dtype\)
            KV_shared_0_r = T\.alloc_shared\(\[(.+?),\s*(.+?)\], dtype\)
            KV_shared_1_l = T\.alloc_shared\(\[(.+?),\s*(.+?)\], dtype\)
            KV_shared_1_r = T\.alloc_shared\(\[(.+?),\s*(.+?)\], dtype\)
            K_tail_shared_0 = T\.alloc_shared\(\[(.+?),\s*(.+?)\], dtype\)
            K_tail_shared_1 = T\.alloc_shared\(\[(.+?),\s*(.+?)\], dtype\)'''

    def alloc_repl(m):
        return f'''            num_stages = 2
            KV_shared_l = T.alloc_shared([num_stages, {m.group(1)}, {m.group(2)}], dtype)
            KV_shared_r = T.alloc_shared([num_stages, {m.group(3)}, {m.group(4)}], dtype)
            K_tail_shared = T.alloc_shared([num_stages, {m.group(9)}, {m.group(10)}], dtype)'''

    content = re.sub(alloc_pattern, alloc_repl, content)

    # remove old num_stages = 2
    content = re.sub(r'            num_stages = 2\n            bar_q = ', r'            bar_q = ', content)

    # 2. tx >= 128 and tx < 256
    c1_pattern = r'''            elif tx >= 128 and tx < 256:
                T\.set_max_nreg\(168, 1\)
                T\.fill\(acc_o_r, 0\)
                for i_i in T\.serial\(T\.ceildiv\(NI, 2\)\):
                    # Buffer 0
                    T\.barrier_arrive\(bar_sScale_and_sS_ready\)
                    T\.barrier_wait\(bar_sScale_and_sS_ready, \(\(i_i \* 2\) & 1\)\)
                    for h_i, d_i in T\.Parallel\((.+?), (.+?)\):
                        acc_o_r\[h_i, d_i\] \*= alpha_shared\[h_i\]
                    T\.gemm\(S_shared, KV_shared_0_r, acc_o_r\)
                    T\.barrier_arrive\(bar_k_free\[0\]\)
                    T\.barrier_arrive\(bar_sScale_and_sS_free\)

                    # Buffer 1
                    T\.barrier_arrive\(bar_sScale_and_sS_ready\)
                    T\.barrier_wait\(bar_sScale_and_sS_ready, \(\(i_i \* 2 \+ 1\) & 1\)\)
                    for h_i, d_i in T\.Parallel\((.+?), (.+?)\):
                        acc_o_r\[h_i, d_i\] \*= alpha_shared\[h_i\]
                    T\.gemm\(S_shared, KV_shared_1_r, acc_o_r\)
                    T\.barrier_arrive\(bar_k_free\[1\]\)
                    if i_i != T\.ceildiv\(NI, 2\) - 1:
                        T\.barrier_arrive\(bar_sScale_and_sS_free\)'''
    def c1_repl(m):
        H_var, D_var = m.group(1), m.group(2)
        return f'''            elif tx >= 128 and tx < 256:
                T.set_max_nreg(168, 1)
                T.fill(acc_o_r, 0)
                for i_i in T.serial(NI):
                    stage = i_i % num_stages
                    phase = (i_i // num_stages) & 1

                    T.barrier_arrive(bar_sScale_and_sS_ready[stage])
                    T.barrier_wait(bar_sScale_and_sS_ready[stage], phase)
                    for h_i, d_i in T.Parallel({H_var}, {D_var}):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_r[stage, :, :], acc_o_r)
                    T.barrier_arrive(bar_k_free[stage])
                    if i_i != NI - 1:
                        T.barrier_arrive(bar_sScale_and_sS_free[stage])'''
    content = re.sub(c1_pattern, c1_repl, content)

    # 3. main_split tx < 128 consumer unroll
    c2_pattern = r'''                for i_i in T\.serial\(T\.ceildiv\(NI, 2\)\):
                    # Buffer 0
                    T\.barrier_wait\(bar_k_ready\[0\], \(i_i & 1\)\)

                    T\.clear\(acc_s\)
                    T\.wgmma_gemm\(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True\)
                    T\.wgmma_gemm\(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True\)
                    T\.wgmma_gemm\(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True\)

                    T\.wait_wgmma\(0\)

                    if i_i != 0:
                        T\.barrier_arrive\(bar_sScale_and_sS_free\)
                        T\.barrier_wait\(bar_sScale_and_sS_free, \(\(i_i \* 2\) & 1\) \^ 1\)

                    T\.copy\(m_i, m_i_prev\)
                    T\.reduce_max\(acc_s, m_i, dim=1, clear=False\)
                    for h_i in T\.Parallel\(block_H\):
                        m_i\[h_i\] = T\.max\(m_i\[h_i\], m_i_prev\[h_i\]\)
                    for h_i in T\.Parallel\(block_H\):
                        alpha_local\[h_i\] = T\.exp2\(\(m_i_prev\[h_i\] - m_i\[h_i\]\) \* sm_scale\)
                    for h_i, bi_i in T\.Parallel\(block_H, block_N\):
                        acc_s\[h_i, bi_i\] = T\.exp2\(acc_s\[h_i, bi_i\] \* sm_scale - m_i\[h_i\] \* sm_scale\)
                    T\.reduce_sum\(acc_s, sumexp_i, dim=1\)  # is this a accumulate operator\?
                    for h_i in T\.Parallel\(block_H\):
                        sumexp\[h_i\] = sumexp\[h_i\] \* alpha_local\[h_i\] \+ sumexp_i\[h_i\]
                    for h_i, d_i in T\.Parallel\(block_H, dim // 2\):
                        acc_o_l\[h_i, d_i\] \*= alpha_local\[h_i\]
                    T\.copy\(alpha_local, alpha_shared\)

                    T\.copy\(acc_s, S_shared\)
                    T\.gemm\(S_shared, KV_shared_0_l, acc_o_l\)

                    T\.barrier_arrive\(bar_sScale_and_sS_ready\)
                    T\.barrier_arrive\(bar_k_free\[0\]\)

                    # Buffer 1
                    T\.barrier_wait\(bar_k_ready\[1\], \(i_i & 1\)\)

                    T\.clear\(acc_s\)
                    T\.wgmma_gemm\(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True\)
                    T\.wgmma_gemm\(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True\)
                    T\.wgmma_gemm\(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True\)

                    T\.wait_wgmma\(0\)

                    T\.barrier_arrive\(bar_sScale_and_sS_free\)
                    T\.barrier_wait\(bar_sScale_and_sS_free, \(\(i_i \* 2 \+ 1\) & 1\) \^ 1\)

                    T\.copy\(m_i, m_i_prev\)
                    T\.reduce_max\(acc_s, m_i, dim=1, clear=False\)
                    for h_i in T\.Parallel\(block_H\):
                        m_i\[h_i\] = T\.max\(m_i\[h_i\], m_i_prev\[h_i\]\)
                    for h_i in T\.Parallel\(block_H\):
                        alpha_local\[h_i\] = T\.exp2\(\(m_i_prev\[h_i\] - m_i\[h_i\]\) \* sm_scale\)
                    for h_i, bi_i in T\.Parallel\(block_H, block_N\):
                        acc_s\[h_i, bi_i\] = T\.exp2\(acc_s\[h_i, bi_i\] \* sm_scale - m_i\[h_i\] \* sm_scale\)
                    T\.reduce_sum\(acc_s, sumexp_i, dim=1\)  # is this a accumulate operator\?
                    for h_i in T\.Parallel\(block_H\):
                        sumexp\[h_i\] = sumexp\[h_i\] \* alpha_local\[h_i\] \+ sumexp_i\[h_i\]
                    for h_i, d_i in T\.Parallel\(block_H, dim // 2\):
                        acc_o_l\[h_i, d_i\] \*= alpha_local\[h_i\]
                    T\.copy\(alpha_local, alpha_shared\)

                    T\.copy\(acc_s, S_shared\)
                    T\.gemm\(S_shared, KV_shared_1_l, acc_o_l\)

                    T\.barrier_arrive\(bar_sScale_and_sS_ready\)
                    T\.barrier_arrive\(bar_k_free\[1\]\)'''
    c2_repl = r'''                for i_i in T.serial(NI):
                    stage = i_i % num_stages
                    phase = (i_i // num_stages) & 1

                    T.barrier_wait(bar_k_ready[stage], phase)

                    T.clear(acc_s)
                    T.wgmma_gemm(Q_shared_l, KV_shared_l[stage, :, :], acc_s, transpose_B=True)
                    T.wgmma_gemm(Q_shared_r, KV_shared_r[stage, :, :], acc_s, transpose_B=True)
                    T.wgmma_gemm(Q_tail_shared, K_tail_shared[stage, :, :], acc_s, transpose_B=True)

                    T.wait_wgmma(0)

                    if i_i != 0:
                        T.barrier_arrive(bar_sScale_and_sS_free[stage])
                        T.barrier_wait(bar_sScale_and_sS_free[stage], phase ^ 1)

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(block_H):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(block_H):
                        alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(block_H, block_N):
                        acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                    T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                    for h_i in T.Parallel(block_H):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(block_H, dim // 2):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)

                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_l[stage, :, :], acc_o_l)

                    T.barrier_arrive(bar_sScale_and_sS_ready[stage])
                    T.barrier_arrive(bar_k_free[stage])'''
    content = re.sub(c2_pattern, c2_repl, content)

    with open(filename, "w") as f:
        f.write(content)

for f in ["examples/deepseek_v32/sparse_mla_fwd_pipelined.py", "examples/deepseek_mla/example_mla_decode_ws.py"]:
    process_file(f)
