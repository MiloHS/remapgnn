# Results summary

## Main result

The current best model is an IRNO-style iterative corrector, v18, trained on top of a frozen v16 conservative remapping operator.

The v18 corrector improves the operator progressively:

| Step | Meaning |
|---|---|
| 0 | frozen v16 base operator |
| 1 | corrected with lmax=8 conditioning |
| 2 | corrected with lmax=16 conditioning |
| 3 | corrected with lmax=24 conditioning |

## Field trajectory

Across six mesh pairs, v18 seed123 improved mean field relative L2 versus TempestRemap as:

| Step | Mean field error |
|---|---:|
| base | 0.002956 |
| corrected lmax=8 | 0.002814 |
| corrected lmax=16 | 0.002733 |
| corrected lmax=24 | 0.002715 |

This is about an 8% improvement over the frozen v16 base.

A second seed reproduced the behavior:

| Step | Mean field error, seed456 |
|---|---:|
| base | 0.002956 |
| corrected lmax=8 | 0.002824 |
| corrected lmax=16 | 0.002758 |
| corrected lmax=24 | 0.002745 |

## Spectral trajectory

The spectral evaluation uses spherical harmonic test fields and compares learned remap outputs against TempestRemap outputs.

For v18 seed123, mean spectral error improved as:

| Step | Mean spectral error |
|---|---:|
| base | 1.583641e-02 |
| corrected lmax=8 | 1.503359e-02 |
| corrected lmax=16 | 1.449893e-02 |
| corrected lmax=24 | 1.425947e-02 |

This is about a 10% improvement in average spectral error.

A second seed also improved monotonically:

| Step | Mean spectral error, seed456 |
|---|---:|
| base | 1.583641e-02 |
| corrected lmax=8 | 1.512504e-02 |
| corrected lmax=16 | 1.473847e-02 |
| corrected lmax=24 | 1.461824e-02 |

## Conclusion

The main scientific result is that a frozen conservative GNN/Sinkhorn remapper can be improved by a shared conditional iterative corrector. The corrector refines the operator progressively across spectral bands while Sinkhorn balancing preserves conservative structure.

