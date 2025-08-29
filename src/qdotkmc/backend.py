from contextlib import nullcontext
import os
from dataclasses import dataclass
from typing import Optional, Dict, Literal, List

class CPUStreams:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def synchronize(self): pass


class Backend:
    """
    Thin CPU/GPU facade over numpy/cupy with a few convenience helpers.

    Attributes
    ----------
    xp : module
        numpy (CPU) or cupy (GPU).
    is_gpu : bool
        True if xp is cupy and a CUDA device is available.
    use_gpu : bool
        Alias of is_gpu for backward compatibility.
    f, c : dtype
        Default real/complex dtypes per `use_c64` policy.
    Stream : context manager
        cupy.cuda.Stream on GPU, no-op stream on CPU.
    """
    def __init__(self, xp, *, use_c64=False, enable_streams=True):
        self.xp = xp
        self.is_gpu = hasattr(xp, "__name__") and xp.__name__.startswith("cupy")
        self.use_gpu = self.is_gpu  # alias to match older call sites
        self.gpu_use_c64 = use_c64

        # dtype policy
        self.f = xp.float32   if use_c64 else xp.float64
        self.c = xp.complex64 if use_c64 else xp.complex128

        # intialize parallel plan attribute
        self.plan = None

        # streams
        # TODO : what do we need this for?
        if self.is_gpu and enable_streams:
            import cupy as cp
            self.Stream = cp.cuda.Stream
        else:
            self.Stream = CPUStreams

        # memory pools (GPU only)
        self.cp = None
        self._kern_cache: Dict[str, object] = {}
        if self.is_gpu:
            import cupy as cp
            self.cp = cp
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            try:
                cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
            except Exception:
                pass


    # ------- array helpers (CPU <-> GPU) -------
    def rawkernel(self, func_name: str, src: str):
        """Return a cached cp.RawKernel; compile once per Backend."""
        if not self.use_gpu or self.cp is None:
            raise RuntimeError("rawkernel requested but GPU backend is not active")
        ker = self._kern_cache.get(func_name)
        if ker is None:
            ker = self.cp.RawKernel(src, func_name)
            self._kern_cache[func_name] = ker
        return ker

    def asarray(self, a, dtype=None, order="C"):
        """Like xp.asarray with sane default dtype (real→f, complex→c)."""
        return self.xp.asarray(
            a,
            dtype=(dtype or (self.c if self._is_complex(a) else self.f)),
            order=order
        )

    def asarray_f(self, a): return self.xp.asarray(a, dtype=self.f, order="C")
    def asarray_c(self, a): return self.xp.asarray(a, dtype=self.c, order="C")

    def from_host(self, a, dtype=None, order="C"):
        """Host→backend array (NumPy no-op; CuPy upload)."""
        return self.xp.asarray(a, dtype=dtype, order=order)

    def to_host(self, a):
        """Backend→host array (CuPy download; NumPy no-op)."""
        if self.is_gpu and self.cp is not None:
            return self.cp.asnumpy(a)
        return a

    def empty(self, shape, dtype=None): return self.xp.empty(shape, dtype=(dtype or self.f))
    def zeros(self, shape, dtype=None): return self.xp.zeros(shape, dtype=(dtype or self.f))
    def conj(self, a): return self.xp.conj(a)
    def abs2(self, a): return self.xp.abs(a)**2
    def matmul(self, A, B): return A @ B
    def sum_axis0(self, a): return a.sum(axis=0)
    def tensordot(self, a, b, axes=2): return self.xp.tensordot(a, b, axes=axes)
    def einsum(self, subscripts, *ops, optimize=False): return self.xp.einsum(subscripts, *ops, optimize=optimize)

    def sync(self):
        """Device synchronize (GPU); no-op on CPU."""
        if self.is_gpu:
            import cupy as cp
            cp.cuda.runtime.deviceSynchronize()

    @staticmethod
    def _is_complex(a):
        import numpy as _np
        return _np.asarray(a).dtype.kind == 'c'


@dataclass(frozen=True)
class ParallelPlan:

    context: Literal["fork", "spawn"]        # mp start method (different for GPU?CPU execution)
    n_workers: int                           # processes to launch
    device_ids: Optional[List[int]]          # GPU ids or None for CPU
    use_gpu: bool                            # whether GPU is intended


# return available CPUs for parallel execution in CPU mode
def _slurm_cpus_per_task(default : int = 1) -> int:
    try:
        return max(1, int(os.getenv("SLURM_CPUS_PER_TASK", str(default))))
    except Exception:
        return default


# recommend plan for parallel execution
def _recommend_parallel_plan(*,
                    use_gpu: bool,
                    do_parallel: bool,
                    max_workers: Optional[int]
                    ) -> ParallelPlan:
    
    # (1) serial execution
    if not do_parallel:
        return ParallelPlan(context="fork", n_workers=1, device_ids=None, use_gpu=use_gpu)
    
    # (2) parallel execution
    # (a) CPU path
    if not use_gpu:
        # match number of workers to SLURM environment
        nw = _slurm_cpus_per_task(1)
        return ParallelPlan(context="fork", n_workers=max(1, nw), device_ids=None, use_gpu=False)
    
    # (b) GPU path
    try:
        import cupy as cp
        n_gpus = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        n_gpus = 1
    # match number of workers to number of availbale GPUs
    # TODO : edit the following so that we do not need max_workers anymore
    nw = max_workers if max_workers is not None else n_gpus
    nw = max(1, min(nw, n_gpus))
    return ParallelPlan(context="spawn", n_workers=nw, device_ids=list(range(nw)), use_gpu=True)


# configure CuBLAS environment for GPU path to enable replicability
def _configure_cublas_env(*, deterministic: Optional[bool] = None,
                          allow_tf32: Optional[bool] = None):
    """
    Configure cuBLAS behavior via environment variables.
    deterministic=True  -> set CUBLAS_WORKSPACE_CONFIG for deterministic kernels
    allow_tf32=True     -> enable TF32 (faster, slightly less accurate FP32 GEMM)
    """
    if deterministic is True:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if allow_tf32 is True:
        os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "1")
    elif allow_tf32 is False:
        os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")


def get_backend(*, 
                prefer_gpu: bool = True, 
                use_c64: bool = False,
                do_parallel: bool = True,
                max_workers: Optional[int] = None
                ):
    """
    returns a Backend bound to CuPy (GPU) if requested else available, else NumPy (CPU).
    attaches ParallelPlan to steer parallel execution (if desired).
    """
    # (1) choose xp (numpy or cupy)
    xp = None
    if prefer_gpu:
        try:
            _configure_cublas_env(deterministic=True, allow_tf32=False)
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                xp = cp
        except Exception:
            xp = None
    if xp is None:
        import numpy as np
        xp = np
    
    # (2) build Backend as before
    be = Backend(xp, use_c64=use_c64)

    # (3) compute and attach parallel plan
    plan = _recommend_parallel_plan(use_gpu=be.use_gpu,
                                    do_parallel=do_parallel,
                                    max_workers=max_workers
                                    )
    be.plan = plan 

    return be