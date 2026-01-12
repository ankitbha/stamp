# =============================================================================
# Scheduled sampling
# =============================================================================

def scheduled_sampling_prob(epoch: int, ss_start_p: float, ss_end_p: float, warmup_epochs: int) -> float:
    e = int(epoch)
    warm = int(warmup_epochs)
    if warm <= 0:
        return float(ss_end_p)
    if e <= 0:
        return float(ss_start_p)
    if e >= warm:
        return float(ss_end_p)
    # linear ramp
    alpha = e / float(warm)
    return float(ss_start_p + alpha * (ss_end_p - ss_start_p))