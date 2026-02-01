# Medium-Voltage Arc-Flash Data Analysis

Tom Short, tshort@epri.com

This code base is using the Julia Language (https://julialang.org/).

To (locally) reproduce this project, do the following:

0. Download this code. 
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and everything should work out of the box.

The main code is in:

* `paper-analysis.jl`

The data is in three CSV files.
These files contain measurements and predictions of incident energies. Each row of the file is one test. Key variables in each file are shown.

1. `epri.csv` -- EPRI tests of MV equipment with predictions based on IEEE 1584-2018
  * `V` -- system voltage, kV (L-L)
  * `Ib` -- bolted fault current, kA
  * `gap_mm` -- electrode gap, mm
  * `D_mm` -- working distance, mm
  * `t` -- duration, msec
  * `Ia` -- predicted arcing current, kA
  * `IE` -- predicted incident energy, cal/cm²
  * `config` -- test configuration (EPRI HCB, EPRI Transformer, EPRI VCB, ...)
  * `Iameas` -- measured average arcing current, kA
  * `IEmeas` -- measured incident energy, cal/cm²
  * `joules` -- arc energy, J

2. `ieeeall.csv` -- Test data from IEEE/NFPA used to develop IEEE 1584-2018
  * `Lab` -- test site
  * `config` -- VCB, HCB, ...
  * `Voc` -- system voltage, kV (L-L)
  * `Ibf` -- bolted fault current, kA
  * `gap_mm` -- electrode gap, mm
  * `D_mm` -- working distance, mm
  * `t` -- duration, msec
  * `width_in` -- box width, in
  * `height_in` -- box height, in
  * `depth_in` -- box depth, in
  * `Iameas` -- measured average arcing current, kA
  * `IEmeas` -- measured incident energy, cal/cm²

3. `hcb.csv` -- Combined EPRI and IEEE/NFPA dataset for HCB configurations
  * `Ib` -- bolted fault  current, kA
  * `gap_mm` -- electrode gap, mm
  * `d_mm` -- working distance, mm
  * `t` -- duration, msec
  * `height_in` -- box height, in
  * `config` -- test configuration (EPRI HCB or EPRI VCB)
  * `iameas` -- measured arcing current, kA
  * `iemeas` -- incident energy measured, cal/cm²
  * `joules` -- arc energy, J
  * `energyratio` -- ratio of incident energy to arc energy, cal/cm²/MJ



