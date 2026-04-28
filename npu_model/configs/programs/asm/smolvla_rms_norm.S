lui x1, 0x2
lui x2, 0x3
lui x3, 0x4
lui x4, 0x5
addi x5, x0, 0
addi x6, x5, 1024
addi x7, x6, 1024
addi x8, x7, 1024
lui x9, 0x1
addi x10, x9, 1024
addi x13, x0, 1024
addi x11, x1, 1024
addi x12, x2, 1024
addi x14, x3, 1024
addi x15, x4, 1024
dma.config.ch0
dma.config.ch1
dma.config.ch2
dma.config.ch3
dma.load.ch0 x1, x5, x13
dma.load.ch1 x11, x6, x13
dma.load.ch2 x2, x7, x13
dma.load.ch3 x12, x7, x13
dma.wait.ch0
dma.wait.ch1
dma.wait.ch2
dma.wait.ch3
dma.load.ch0 x3, x8, x13
dma.load.ch1 x14, x8, x13
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
vload m6, 0(x2)
delay 34
vload m7, 32(x2)
delay 34
vsquare.bf16 m2, m0
delay 66
vredsum.row.bf16 m4, m2
delay 39
vmul.bf16 m10, m4, m6
delay 66
dma.wait.ch0
dma.wait.ch1
vload m8, 0(x3)
delay 34
vload m9, 32(x3)
delay 34
vadd.bf16 m12, m10, m8
delay 66
vsqrt.bf16 m14, m12
delay 66
vrecip.bf16 m4, m14
delay 66
vmul.bf16 m6, m0, m4
delay 66
vstore m6, 0(x4)
delay 34
vstore m7, 32(x4)
delay 34
dma.store.ch0 x9, x4, x13
dma.store.ch1 x10, x15, x13
dma.wait.ch0
dma.wait.ch1
