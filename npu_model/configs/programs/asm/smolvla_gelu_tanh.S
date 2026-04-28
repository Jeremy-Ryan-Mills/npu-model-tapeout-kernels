lui x1, 0x2
lui x2, 0x3
lui x3, 0x4
lui x4, 0x5
lui x5, 0x6
addi x6, x0, 0
addi x7, x6, 1024
lui x8, 0x1
addi x8, x8, -2048
addi x9, x8, 1024
lui x10, 0x1
lui x11, 0x1
addi x11, x11, 1024
addi x12, x11, 1024
addi x13, x0, 1024
addi x14, x1, 1024
addi x18, x5, 1024
dma.config.ch0
dma.wait.ch0
dma.load.ch0 x1, x6, x13
dma.load.ch1 x14, x7, x13
dma.load.ch2 x2, x8, x13
dma.load.ch3 x3, x9, x13
dma.wait.ch0
dma.load.ch0 x4, x10, x13
dma.wait.ch1
vload m0, 0(x1)
delay 34
vload m1, 32(x1)
delay 34
vmul.bf16 m10, m0, m0
delay 66
vmul.bf16 m12, m10, m0
delay 66
vli.all m8, 1
delay 65
vli.all m9, 1
delay 65
dma.wait.ch2
vload m2, 0(x2)
delay 34
vload m3, 0(x2)
delay 34
vmul.bf16 m14, m12, m2
delay 66
vadd.bf16 m16, m0, m14
delay 66
dma.wait.ch3
vload m4, 0(x3)
delay 34
vload m5, 0(x3)
delay 34
vmul.bf16 m18, m16, m4
delay 66
vtanh.bf16 m20, m18, m0
delay 66
vadd.bf16 m22, m8, m20
delay 66
dma.wait.ch0
vload m6, 0(x4)
delay 34
vload m7, 0(x4)
delay 34
vmul.bf16 m24, m0, m6
delay 66
vmul.bf16 m26, m24, m22
delay 66
vstore m26, 0(x5)
delay 34
dma.store.ch0 x11, x5, x13
vstore m27, 32(x5)
delay 34
dma.store.ch1 x12, x18, x13
dma.wait.ch0
dma.wait.ch1
