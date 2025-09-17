# utilities that help print to log file if desired (verbosity = True)
from .backend import Backend


def simulated_time(sim_time : float) -> str:
    """
    return simulated time
    """
    border = "=" * 42
    header = " REDFIELD RATES TIME SUMMARY "
    return (
        f"{border}\n"
        f"{header.center(42)}\n"
        f"{'-'*42}\n"
        f" Total simulated time : {sim_time:10.3f} s\n"
        f"{border}"
    )

def mean_kmc_steps(mean_steps: float) -> str:
    """
    Return a formatted summary string for mean KMC step count.
    """
    border = "=" * 42
    header = " KMC STEP COUNT SUMMARY "
    return (
        f"{border}\n"
        f"{header.center(42)}\n"
        f"{'-'*42}\n"
        f" Mean KMC steps taken  : {mean_steps:10.1f} steps\n"
        f"{border}"
    )

def backend_summary(backend : Backend) -> str:
    """
    print backend details
    """
    xp_name   = getattr(backend.xp, "__name__", str(backend.xp))
    is_gpu    = getattr(backend, "use_gpu", False)

    # Parallel plan info
    plan       = getattr(backend, "plan", None)
    n_workers  = getattr(plan, "n_workers", None) if plan else None
    context    = getattr(plan, "context", None)   if plan else None
    device_ids = getattr(plan, "device_ids", None) if plan else None

    parallel_mode = "parallel" if (n_workers or 0) > 1 else "serial"

    lines = []
    lines.append("── Backend Summary ──")
    lines.append(f"Backend : {'GPU (CuPy)' if is_gpu else 'CPU (NumPy)'} [{xp_name}]")
    lines.append(f"Mode    : {parallel_mode}")
    if parallel_mode == "parallel":
        lines.append(f"Workers : {n_workers} (context={context})")
    if is_gpu and device_ids:
        lines.append(f"Devices : {', '.join(str(d) for d in device_ids)}")
    lines.append("─────────────────────")

    return "\n".join(lines)