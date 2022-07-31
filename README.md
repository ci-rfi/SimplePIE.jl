# SimplePIE

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chenspc.github.io/SimplePIE.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://chenspc.github.io/SimplePIE.jl/dev/)
[![Build Status](https://github.com/chenspc/SimplePIE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/chenspc/SimplePIE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chenspc/SimplePIE.jl?svg=true)](https://ci.appveyor.com/project/chenspc/SimplePIE-jl)
[![Coverage](https://codecov.io/gh/chenspc/SimplePIE.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/chenspc/SimplePIE.jl)

## Variable names
| Symbol | Name                                                  | Tab completion sequence    |
|--------|-------------------------------------------------------|----------------------------|
| λ      | wavelength                                            | `\lambda` `TAB`            |
| α      | convergence semi-angle                                | `\alpha` `TAB`             |
| D      | diameter of bright field disks in pixel               | `D`                        |
| Δx     | real space sampling                                   | `\Delta` `TAB` `x`         |
| Δk     | reciprocal space sampling                             | `\Delta` `TAB` `k`         |
| N      | detector array size (N x N)                           | `N`                        |
| n      | probe position per line                               | `n`                        |
| n₁     | probe position in direction 1                         | `n\_1` `TAB`               |
| n₂     | probe position in direction 2                         | `n\_2` `TAB`               |
| nᵢ     | number of iteration                                   | `n\_i` `TAB`               |
| θ      | max angle in reciprocal space                         | `\theta` `TAB`             |
| θᵣ     | scan rotation                                         | `\theta` `TAB` `\_r` `TAB` |
| dₛ     | step size                                             | `d\_s` `TAB`               |
| Δf     | defocus                                               | `\Delta` `TAB` `f`         |
| 𝒪      | object                                                | `\scrO` `TAB`              |
| ℴ      | sub-object group conresponding <br>to probe positions | `\scro` `TAB`              |
| 𝒫      | probe                                                 | `\scrP` `TAB`              |
| 𝒜      | amplitute <br>(DPs after sqrt and fftshift)           | `\scrA` `TAB`              |
| 𝒟      | diffraction                                           | `\scrD` `TAB`              |
