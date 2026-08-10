"""Global settings for the Markov-Switching TVTP Monte Carlo study."""

# Monte Carlo replication count (paper uses 50; reduced to 20 for tractable runtime)
R = 20

# Optimisation starts per estimation run (MC) (paper uses 10; reduced to 5 for speed)
N_STARTS = 5

# Sample sizes to evaluate
T_LIST = [500, 1000]

# Burn-in observations discarded from likelihood accumulation
BURN_IN = 100

# End-trim observations discarded from likelihood accumulation
END_TRIM = 10

# Effective likelihood observations: T - BURN_IN - END_TRIM
# {500: 390, 1000: 890}
EFF_OBS = {T: T - BURN_IN - END_TRIM for T in T_LIST}

# Threshold for filtering extreme |A| estimates in DGP 7, T=500, ModelI
EXTREME_A_THRESHOLD = 10.0

# Base seed for replication-level RNG construction
SEED_BASE = 12345

# Empirical application: 762 yield changes − 100 burn-in = 662 effective obs
EFF_OBS_EMP = 662

# Optimisation starts for empirical application (more starts for real data)
N_STARTS_EMP = 20

