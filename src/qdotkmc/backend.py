from contextlib import nullcontext
import os
from typing import Optional, Dict

class CPUStreams:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def synchronize(self): pass

# backend.py
from contextlib import nullcontext
import os
from typing import Optional

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

        # streams
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


    # ------- array helpers (CPU/GPU) -------
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

    # def setup_pools(self):
    #     """(Re)configure memory pools (GPU); no-op on CPU."""
    #     if self.is_gpu:
    #         import cupy as cp
    #         cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    #         try:
    #             cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
    #         except Exception:
    #             pass

    def sync(self):
        """Device synchronize (GPU); no-op on CPU."""
        if self.is_gpu:
            import cupy as cp
            cp.cuda.runtime.deviceSynchronize()

    @staticmethod
    def _is_complex(a):
        import numpy as _np
        return _np.asarray(a).dtype.kind == 'c'


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


def get_backend(*, prefer_gpu=True, use_c64=False):
    """
    Return a Backend bound to CuPy (GPU) if requested & available, else NumPy (CPU).
    Sets cuBLAS env early for deterministic kernels and TF32 policy.
    """
    if prefer_gpu:
        try:
            _configure_cublas_env(deterministic=True, allow_tf32=False)
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                return Backend(cp, use_c64=use_c64)
        except Exception:
            pass
    import numpy as np
    return Backend(np, use_c64=use_c64)