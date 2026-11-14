# Training data: `soil_co2_traces.npz`

300 traces, each 2016 half-hour steps (42 days), from 20 sites over 2023–2025 (5 growth phases per site-year).
The inputs are synthetic site data driven by ERA5 reanalysis weather; the targets come from a soil-carbon process model
run on those inputs. The network learns to emulate the process model's heterotrophic CO₂ flux.

| array | shape, dtype | contents |
|---|---|---|
| `inputs` | (300, 2016, 5) float32 | soil temperature (°C), volumetric soil moisture, surface solar radiation downwards (ERA5 SSRD), air temperature (°C), day of year / 366 |
| `targets` | (300, 2016, 3) float32 | microbial biomass carbon (MBC), soil organic carbon (SOC), heterotrophic CO₂ flux (µmol m⁻² s⁻¹); the model predicts the last |
| `initial_mbc` | (300,) float64 | initial MBC; fed to the model as a constant sixth input channel |
| `colonization_lag_days` | (300,) float64 | days until microbial colonization begins; supervises the colonization detector |
| `site_id`, `year` | (300,) int32 | labels |

The training code fixes the split with seed 42: 240 training, 30 validation and 30 test traces.

**Provenance.** Exported from the team's ERA5 training release `training_data_era5_v5_63c2dc1` (zarr v2; sha256
`b51e08db475938278cc05668cd47a2a533ed0199e73435f5a8b02657a5eec965`), with only the arrays above. Loading this file gives
arrays bit-identical to the original loader's.

sha256 of `soil_co2_traces.npz`: `644d8483a220e22f35802b68f9beb7eb411ca3dadd1b96a6b2eefff122e45027`

**Attribution.** Contains modified Copernicus Climate Change Service information (2026). Neither the European
Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
Hersbach, H. et al. (2020), The ERA5 global reanalysis, *Q. J. R. Meteorol. Soc.* 146, 1999–2049, doi:10.1002/qj.3803.
