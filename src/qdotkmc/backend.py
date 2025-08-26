from contextlib import nullcontext

class CPUStreams:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def synchronize(self): pass

class Backend:
    def __init__(self, xp, *, use_c64=False, enable_streams=True):
        self.xp = xp
        self.is_gpu = hasattr(xp, "__name__") and xp.__name__.startswith("cupy")
        # dtype policy
        self.f = xp.float32  if use_c64 else xp.float64
        self.c = xp.complex64 if use_c64 else xp.complex128

        # streams
        if self.is_gpu and enable_streams:
            import cupy as cp
            self.Stream = cp.cuda.Stream
        else:
            self.Stream = CPUStreams

        # memory pools (GPU only)
        if self.is_gpu:
            import cupy as cp
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            try:
                cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
            except Exception:
                pass

    # array helpers
    def asarray(self, a, dtype=None, order="C"):
        return self.xp.asarray(a, dtype=(dtype or self.c if self._is_complex(a) else self.f), order=order)
    def asarray_f(self, a): return self.xp.asarray(a, dtype=self.f, order="C")
    def asarray_c(self, a): return self.xp.asarray(a, dtype=self.c, order="C")
    def empty(self, shape, dtype=None): return self.xp.empty(shape, dtype=(dtype or self.f))
    def zeros(self, shape, dtype=None): return self.xp.zeros(shape, dtype=(dtype or self.f))
    def conj(self, a): return self.xp.conj(a)
    def abs2(self, a): return self.xp.abs(a)**2
    def matmul(self, A, B): return A @ B
    def sum_axis0(self, a): return a.sum(axis=0)
    def tensordot(self, a, b, axes=2): return self.xp.tensordot(a, b, axes=axes)
    def einsum(self, subscripts, *ops, optimize=False): return self.xp.einsum(subscripts, *ops, optimize=optimize)

    def to_host(self, a):
        if self.is_gpu:
            import cupy as cp
            return cp.asnumpy(a)
        return a

    def sync(self):
        if self.is_gpu:
            import cupy as cp
            cp.cuda.runtime.deviceSynchronize()

    @staticmethod
    def _is_complex(a):
        import numpy as _np
        return _np.asarray(a).dtype.kind == 'c'



def get_backend(*, prefer_gpu=True, use_c64=False):
    """return a Backend instance for GPU if available, else CPU."""
    if prefer_gpu:
        try:
            import cupy as cp
            # try a cheap CUDA call
            ndev = cp.cuda.runtime.getDeviceCount()
            if ndev > 0:
                return Backend(cp, use_c64=use_c64)
        except Exception:
            pass
    import numpy as np
    return Backend(np, use_c64=use_c64)