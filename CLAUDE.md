# Overview

This is the performance model for the NPU. Currently, there are smolvla and parameterizable kernels at npu_model/configs/programs. So far, the smolvla kernels are the most accurate in terms of instruction sequence + delays. Read /npu_spec/ to understand the specifications of the npu in extreme detail.

## Task

Right now, the kernels are kind of a mess, they pass the regression (pass all tests in the /test directory, but the kernels are pretty unoptimized. When they were generated, it was clear the spec wasn't fully taken into account. An example of something that went wrong is that only MXU0 is used in the kernels, not MXU1.

Your task is to first, optimize the smolvla kernels. I want you to keep track of the CPI of the kernels, and improve them by overlapping instructions in the MXU and VPU, pipelining, etc. Try to optimize each kernel reasonably well, and make sure they all still pass. There is one known error in the perf mode. vmatpop.fp8 instructions currently don't use a scaling factor to perform quantization. For now, write the kernels to pass the tests, and leave comments on what to change it to after the fix.

The next task for you is to update the parameterized kernels. These are a little more outdated, and probably don't have hte correct latencies. Additionally, all of the kernels are created by loop unrolling, which can lead to some huge programs. Please change this to jumping. Fix these kernels so that they pass their test with different shapes in the /test/ directory. Also, try optimizing them a bit with software pipelining + other strategies by using both matrices + vpu.

## Rules

No stupid comments, I want all of the smolvla kernels to have a similar commenting style, and dont make it look lika AI. No long ------ across the screen, no special characters, etc. 

Make sure the tests pass, and test them more locally. Test the CPI locally.


## Important Directories

`/npu_model/configs/programs/ - Kernels
`/npu_model/hardware/` - Contains latencies for different vector and matrix operations
`/npu_spec/` - Markdown specification for the NPU
