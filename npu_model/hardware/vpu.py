import math

from .exu import ExecutionUnit
from ..logging.logger import Logger, LaneType
from ..hardware.arch_state import ArchState
from ..software.instruction import Uop
from ..isa import EXU
from .stage_data import StageData
from .config import HardwareConfig
from .bank_conflict import mrf_accesses, vmem_accesses


VPU_SIMPLE_OPS = {
    "vadd.bf16",
    "vsub.bf16",
    "vmul.bf16",
    "vrelu.bf16",
    "vsquare.bf16",
    "vcube.bf16",
    "vminimum.bf16",
    "vmaximum.bf16",
    "vredsum.bf16",
    "vredmin.bf16",
    "vredmax.bf16",
    "vredsum.row.bf16",
    "vredmin.row.bf16",
    "vredmax.row.bf16",
}

VPU_NON_PIPELINEABLE_OPS = {
    "vrecip.bf16",
    "vexp.bf16",
    "vexp2.bf16",
    "vsin.bf16",
    "vcos.bf16",
    "vtanh.bf16",
    "vlog2.bf16",
    "vsqrt.bf16",
    "vlog.bf16",
    "vmax.bf16",
    "vmin.bf16",
}

VPU_LOCAL_TRANSFER_OPS = {
    "vmov",
}

# no specific delay at this point
XLU_OPS = {"vtrpose.xlu"}

LOCAL_TRANSFER_TILE_BYTES = {
    "vmov": 1024,
}


class VectorExecutionUnit(ExecutionUnit):
    """Execution unit for vector operations."""

    def __init__(
        self,
        name: str,
        logger: Logger,
        arch_state: ArchState,
        lane_id: int = 0,
        config: HardwareConfig | None = None,
    ) -> None:
        super().__init__(
            name,
            logger,
            arch_state,
            lane_id,
            config,
        )
        self.reset()

    def can_handle(self, uop: Uop) -> bool:
        return True

    def reset(self) -> None:
        self.in_flight: Uop | None = None
        self._in_flight_mrf_banks: frozenset[int] = frozenset()
        self._in_flight_vmem_banks: frozenset[int] = frozenset()
        self._complete_count = 0
        self._pending_completions: list[Uop] = []
        self._total_instructions = 0
        self._busy_cycles = 0

    def _execution_latency(self, uop: Uop) -> int:
        mnemonic = uop.insn.mnemonic
        if mnemonic in LOCAL_TRANSFER_TILE_BYTES:
            return max(
                1,
                math.ceil(
                    LOCAL_TRANSFER_TILE_BYTES[mnemonic]
                    / self.config.vmem_bytes_per_cycle
                ),
            )
        if mnemonic in XLU_OPS:
            return self.config.xlu_transform_latency_cycles
        if mnemonic in VPU_LOCAL_TRANSFER_OPS:
            return self.config.vpu_simple_op_latency_cycles
        if mnemonic in VPU_SIMPLE_OPS:
            return self.config.vpu_simple_op_latency_cycles
        if mnemonic in VPU_NON_PIPELINEABLE_OPS:
            return self.config.vpu_non_pipelineable_op_latency_cycles
        else:
            return 1

    def tick(self, idu_output: StageData[Uop | None]) -> None:
        self.cycle += 1
        # Log deferred completions from last cycle
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id, cycle=self.cycle)
            self.logger.log_retire(uop.id)

        self._pending_completions = []

        self._complete_count = 0

        if self.in_flight is None:
            # Peek instruction from DIU
            uop = None
            if self.in_flight is None:
                uop = idu_output.peek()

            # Accept new instruction
            if uop is not None:
                assert uop.insn.exu == EXU.VECTOR, "Non-vector instruction passed to Vector Unit."
                label = f"{self.name}:{uop.insn.mnemonic}"
                mrf_banks = mrf_accesses(uop.insn)
                vmem_banks = vmem_accesses(uop.insn, self.arch_state)
                checker = self.arch_state.conflict_checker
                checker.acquire_mrf(mrf_banks, label)
                checker.acquire_vmem(vmem_banks, label)
                self._in_flight_mrf_banks = mrf_banks
                self._in_flight_vmem_banks = vmem_banks
                # tag instruction with execution delay
                uop.execute_delay = self._execution_latency(uop)
                self.in_flight = uop
                self._total_instructions += 1
                # Log: end dispatch, start execute
                self.logger.log_stage_end(
                    uop.id,
                    "D",
                    lane=LaneType.DIU.value,
                    cycle=self.cycle,
                )
                self.logger.log_stage_start(
                    uop.id,
                    "E",
                    lane=self.lane_id,
                    cycle=self.cycle,
                )

        # Track if EXU was busy
        if self.is_busy():
            self._busy_cycles += 1

        # Process in-flight instructions
        if self.in_flight:
            self.in_flight.execute_delay -= 1
            if self.in_flight.execute_delay <= 0:
                # execute the instruction
                self.in_flight.insn.exec(self.arch_state)
                self._complete_count = 1
                # Release acquired banks before retiring the instruction.
                checker = self.arch_state.conflict_checker
                checker.release_mrf(self._in_flight_mrf_banks)
                checker.release_vmem(self._in_flight_vmem_banks)
                self._in_flight_mrf_banks = frozenset[int]()
                self._in_flight_vmem_banks = frozenset[int]()
                # Defer completion logging to next tick
                self._pending_completions.append(self.in_flight)
                # claim the uop from the DIU
                idu_output.claim()
                self.in_flight = None

    def flush_completions(self) -> None:
        """Flush any pending completions (call at end of simulation)."""
        for uop in self._pending_completions:
            self.logger.log_stage_end(uop.id, "E", lane=self.lane_id)
            self.logger.log_retire(uop.id)
        self._pending_completions = []

    def is_busy(self) -> bool:
        """Check if the EXU is busy."""
        return self.in_flight is not None and self.in_flight.insn.mnemonic != "delay"

    @property
    def has_in_flight(self) -> bool:
        """Check if there are any in-flight instructions."""
        return self.in_flight is not None

    @property
    def complete_count(self) -> int:
        """Instructions completed this cycle."""
        return self._complete_count

    @property
    def total_instructions(self) -> int:
        """Total instructions executed."""
        return self._total_instructions

    @property
    def busy_cycles(self) -> int:
        """Number of cycles the EXU was busy."""
        return self._busy_cycles
