# SmolVLA fused attention kernel (flash attention)
# output = softmax(Q @ K^T * scale) @ V
# One Q-block (32 rows), two K tiles (k_seq=64), head_dim=64.
# Inputs bf16; quantized to fp8 on-chip via acc roundtrip.
#
# Scalar register map:
#   VMEM:  x1=Q  x2=KT0  x3=KT1  x4=VT0  x5=VT1  x6=SCALE  x7=OUT
#   DRAM:  x0=Q  x9=KT0  x10=KT1  x11=VT0  x12=VT1  x13=SCALE  x14=OUT
#   Sizes: x15=4096 (tile bytes)  x16=1024 (scale bytes)
#
# MRF register map:
#   Persistent:
#     (m0,m1)   = Q_lo bf16     (m2,m3)   = Q_hi bf16
#     m4        = Q_lo fp8      m5        = Q_hi fp8
#     (m6,m7)   = scale (broadcast)
#     (m8,m9)   = m_prev (running row-max; -100 as -inf proxy)
#     (m10,m11) = l_prev (running row-sum; init 0)
#     (m12,m13) = O_left  (output cols 0:32)
#     (m14,m15) = O_right (output cols 32:64)
#   Per K-tile (overwritten each iteration):
#     (m16,m17) = KT_top bf16   m20 = KT_top fp8
#     (m18,m19) = KT_bot bf16   m22 = KT_bot fp8
#     (m24,m25) = V_left bf16   m28 = V_left fp8
#     (m26,m27) = V_right bf16  m30 = V_right fp8
#   Temporaries:
#     (m32,m33) = scores bf16      (m34,m35) = scaled bf16
#     (m36,m37) = tile_max / exp_diff bf16
#     (m38,m39) = m_new bf16       (m40,m41) = exp_s bf16
#     m42       = exp_s fp8        (m44,m45) = vc_left bf16
#     (m46,m47) = vc_right bf16    (m48,m49) = exp_diff*l / inv_l
#     (m50,m51) = rowsum(exp_s)

# --- Scalar setup: VMEM addresses ---
lui x1, 0x8                 # x1  = 0x8000 VMEM_Q
lui x2, 0x9                 # x2  = 0x9000 VMEM_KT0
lui x3, 0xA                 # x3  = 0xA000 VMEM_KT1
lui x4, 0xB                 # x4  = 0xB000 VMEM_VT0
lui x5, 0xC                 # x5  = 0xC000 VMEM_VT1
lui x6, 0xD                 # x6  = 0xD000 VMEM_SCALE
lui x7, 0xE                 # x7  = 0xE000 VMEM_OUT

# --- Scalar setup: DRAM addresses + transfer sizes ---
# x0 = 0x0000 DRAM_Q (hardwired zero)
lui x9, 0x1                 # x9  = 0x1000 DRAM_KT0
lui x10, 0x2                # x10 = 0x2000 DRAM_KT1
lui x11, 0x3                # x11 = 0x3000 DRAM_VT0
lui x12, 0x4                # x12 = 0x4000 DRAM_VT1
lui x13, 0x5                # x13 = 0x5000 DRAM_SCALE
lui x14, 0x6                # x14 = 0x6000 DRAM_OUT
lui x15, 0x1                # x15 = 0x1000 = 4096 (tile bytes)
addi x16, x0, 1024          # x16 = 1024 (scale bytes)

# --- DMA: DRAM -> VMEM ---
dma.config.ch0 x0
dma.wait.ch0
dma.load.ch0 x1, x0, x15   # Q
dma.load.ch1 x2, x9, x15   # KT0
dma.load.ch2 x3, x10, x15  # KT1
dma.load.ch3 x4, x11, x15  # VT0
dma.wait.ch0
dma.wait.ch1
dma.wait.ch2
dma.wait.ch3
dma.load.ch0 x5, x12, x15  # VT1
dma.load.ch1 x6, x13, x16  # SCALE
dma.wait.ch0
dma.wait.ch1

# unit scale for all bf16->fp8 acc roundtrips
seli e0, 1

# --- Q: VMEM -> MRF (4 x [32,16] blocks = [32,64]) ---
vload m0, 0(x1)             # Q_lo h0
delay 34
vload m1, 32(x1)            # Q_lo h1
delay 34
vload m2, 64(x1)            # Q_hi h0
delay 34
vload m3, 96(x1)            # Q_hi h1
delay 34

# --- Q_lo (m0,m1) bf16 -> m4 fp8 via acc[1] roundtrip ---
vmatpush.acc.bf16.mxu0 acc1, m0
delay 32
vmatpop.fp8.acc.mxu0 m4, acc1, e0
delay 32

# --- Q_hi (m2,m3) bf16 -> m5 fp8 ---
vmatpush.acc.bf16.mxu0 acc1, m2
delay 32
vmatpop.fp8.acc.mxu0 m5, acc1, e0
delay 32

# --- Scale: load into (m6, m7) (same 1024-B block broadcast to both halves) ---
vload m6, 0(x6)
delay 34
vload m7, 0(x6)
delay 34

# --- Initialize online-softmax state ---
vli.all m8, -100            # m_prev h0 (-100 as -inf proxy)
delay 65
vli.all m9, -100            # m_prev h1
delay 65
vli.all m10, 0              # l_prev h0
delay 65
vli.all m11, 0              # l_prev h1
delay 65
vli.all m12, 0              # O_left h0
delay 65
vli.all m13, 0              # O_left h1
delay 65
vli.all m14, 0              # O_right h0
delay 65
vli.all m15, 0              # O_right h1
delay 65

# ======================================================================
# K TILE 0  (k_seq 0:32)
# ======================================================================

# --- Load KT0: K^T[64,32] column-blocked -> (m16,m17)=KT_top, (m18,m19)=KT_bot ---
vload m16, 0(x2)
delay 34
vload m17, 32(x2)
delay 34
vload m18, 64(x2)
delay 34
vload m19, 96(x2)
delay 34

# --- KT_top (m16,m17) bf16 -> m20 fp8 ---
vmatpush.acc.bf16.mxu0 acc1, m16
delay 32
vmatpop.fp8.acc.mxu0 m20, acc1, e0
delay 32

# --- KT_bot (m18,m19) bf16 -> m22 fp8 ---
vmatpush.acc.bf16.mxu0 acc1, m18
delay 32
vmatpop.fp8.acc.mxu0 m22, acc1, e0
delay 32

# --- Load VT0: V_std[32,64] column-blocked -> (m24,m25)=V_left, (m26,m27)=V_right ---
vload m24, 0(x4)
delay 34
vload m25, 32(x4)
delay 34
vload m26, 64(x4)
delay 34
vload m27, 96(x4)
delay 34

# --- V_left (m24,m25) bf16 -> m28 fp8 ---
vmatpush.acc.bf16.mxu0 acc1, m24
delay 32
vmatpop.fp8.acc.mxu0 m28, acc1, e0
delay 32

# --- V_right (m26,m27) bf16 -> m30 fp8 ---
vmatpush.acc.bf16.mxu0 acc1, m26
delay 32
vmatpop.fp8.acc.mxu0 m30, acc1, e0
delay 32

# --- Q @ K^T: acc[0] = Q_lo @ KT_top + Q_hi @ KT_bot ---
vmatpush.weight.mxu0 w0, m20
delay 32
vmatmul.mxu0 acc0, m4, w0
delay 96
vmatpush.weight.mxu0 w0, m22
delay 32
vmatmul.acc.mxu0 acc0, m5, w0
delay 96
vmatpop.bf16.acc.mxu0 m32, acc0    # (m32, m33) = scores
delay 32

# --- scaled = scores * scale ---
vmul.bf16 m34, m32, m6
delay 66

# --- tile_max = rowmax(scaled) ---
vredmax.row.bf16 m36, m34
delay 69

# --- m_new = max(m_prev, tile_max) ---
vmaximum.bf16 m38, m8, m36
delay 69

# --- exp_diff = exp(m_prev - m_new) ---
vsub.bf16 m36, m8, m38
delay 66
vexp.bf16 m36, m36
delay 66

# --- O *= exp_diff ---
vmul.bf16 m12, m12, m36     # O_left
delay 66
vmul.bf16 m14, m14, m36     # O_right
delay 66

# --- exp_s = exp(scaled - m_new) ---
vsub.bf16 m40, m34, m38
delay 66
vexp.bf16 m40, m40
delay 66

# --- exp_s (m40,m41) bf16 -> m42 fp8 ---
vmatpush.acc.bf16.mxu0 acc1, m40
delay 32
vmatpop.fp8.acc.mxu0 m42, acc1, e0
delay 32

# --- vc_left = exp_s @ V_left ---
vmatpush.weight.mxu0 w0, m28
delay 32
vmatmul.mxu0 acc0, m42, w0
delay 96
vmatpop.bf16.acc.mxu0 m44, acc0    # (m44, m45)
delay 32

# --- vc_right = exp_s @ V_right ---
vmatpush.weight.mxu0 w0, m30
delay 32
vmatmul.mxu0 acc0, m42, w0
delay 96
vmatpop.bf16.acc.mxu0 m46, acc0    # (m46, m47)
delay 32

# --- O += V contribution ---
vadd.bf16 m12, m12, m44     # O_left
delay 66
vadd.bf16 m14, m14, m46     # O_right
delay 66

# --- l = exp_diff * l + rowsum(exp_s) ---
vmul.bf16 m48, m36, m10
delay 66
vredsum.row.bf16 m50, m40
delay 69
vadd.bf16 m10, m48, m50
delay 66

# --- m_prev = m_new (vmov is single-register; copy both halves explicitly) ---
vmov m8, m38
delay 66
vmov m9, m39
delay 66

# ======================================================================
# K TILE 1  (k_seq 32:64)
# ======================================================================

# --- Load KT1 ---
vload m16, 0(x3)
delay 34
vload m17, 32(x3)
delay 34
vload m18, 64(x3)
delay 34
vload m19, 96(x3)
delay 34
vmatpush.acc.bf16.mxu0 acc1, m16
delay 32
vmatpop.fp8.acc.mxu0 m20, acc1, e0
delay 32
vmatpush.acc.bf16.mxu0 acc1, m18
delay 32
vmatpop.fp8.acc.mxu0 m22, acc1, e0
delay 32

# --- Load VT1 ---
vload m24, 0(x5)
delay 34
vload m25, 32(x5)
delay 34
vload m26, 64(x5)
delay 34
vload m27, 96(x5)
delay 34
vmatpush.acc.bf16.mxu0 acc1, m24
delay 32
vmatpop.fp8.acc.mxu0 m28, acc1, e0
delay 32
vmatpush.acc.bf16.mxu0 acc1, m26
delay 32
vmatpop.fp8.acc.mxu0 m30, acc1, e0
delay 32

# --- Q @ K1^T ---
vmatpush.weight.mxu0 w0, m20
delay 32
vmatmul.mxu0 acc0, m4, w0
delay 96
vmatpush.weight.mxu0 w0, m22
delay 32
vmatmul.acc.mxu0 acc0, m5, w0
delay 96
vmatpop.bf16.acc.mxu0 m32, acc0
delay 32
vmul.bf16 m34, m32, m6
delay 66
vredmax.row.bf16 m36, m34
delay 69
vmaximum.bf16 m38, m8, m36
delay 69
vsub.bf16 m36, m8, m38
delay 66
vexp.bf16 m36, m36
delay 66
vmul.bf16 m12, m12, m36
delay 66
vmul.bf16 m14, m14, m36
delay 66
vsub.bf16 m40, m34, m38
delay 66
vexp.bf16 m40, m40
delay 66
vmatpush.acc.bf16.mxu0 acc1, m40
delay 32
vmatpop.fp8.acc.mxu0 m42, acc1, e0
delay 32
vmatpush.weight.mxu0 w0, m28
delay 32
vmatmul.mxu0 acc0, m42, w0
delay 96
vmatpop.bf16.acc.mxu0 m44, acc0
delay 32
vmatpush.weight.mxu0 w0, m30
delay 32
vmatmul.mxu0 acc0, m42, w0
delay 96
vmatpop.bf16.acc.mxu0 m46, acc0
delay 32
vadd.bf16 m12, m12, m44
delay 66
vadd.bf16 m14, m14, m46
delay 66
vmul.bf16 m48, m36, m10
delay 66
vredsum.row.bf16 m50, m40
delay 69
vadd.bf16 m10, m48, m50
delay 66
vmov m8, m38
delay 66
vmov m9, m39
delay 66

# --- Normalize: O /= l ---
vrecip.bf16 m48, m10
delay 66
vmul.bf16 m12, m12, m48     # O_left  /= l
delay 66
vmul.bf16 m14, m14, m48     # O_right /= l
delay 66

# --- MRF -> VMEM -> DRAM (4 x [32,16] blocks = [32,64]) ---
vstore m12, 0(x7)
delay 34
vstore m13, 32(x7)
delay 34
vstore m14, 64(x7)
delay 34
vstore m15, 96(x7)
delay 34
dma.store.ch0 x14, x7, x15
dma.wait.ch0
