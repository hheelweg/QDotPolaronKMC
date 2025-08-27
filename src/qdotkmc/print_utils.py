# utilities that help print to log file if desired (verbosity = True)

def simulated_time(sim_time : float) -> str:
    """
    return simulated time
    """
    border = "=" * 42
    header = " SIMULATED TIME SUMMARY "
    return (
        f"{border}\n"
        f"{header.center(42)}\n"
        f"{'-'*42}\n"
        f" Total simulated time : {sim_time:10.3f} s\n"
        f"{border}"
    )