# Dataset Analysis: Wireless Technology Classification

**Paper:** "A Framework for Wireless Technology Classification using Crowdsensing Platforms"
**Dataset Location:** `dataset/opt/shared/alessio/Data_Transmitters_Identification/spectrum_bands_2/`

---

## Key Definitions (from the Paper)

| Term | Definition |
|------|------------|
| **PSD segment** | A single row of the spectrogram matrix, vector x ∈ R^M where M ≈ 215 bins (~2 MHz bandwidth) |
| **Segment duration** | 870 segments × 215 bins = ~40 seconds → **1 segment ≈ 0.046 seconds** |
| **Hopping strategy** | For bandwidth > 2 MHz, split into 2 MHz chunks sequentially from the first frequency bin |
| **Narrowband** | Bandwidth ≤ 2 MHz: 1 chunk per time segment (use full signal as-is) |
| **Wideband** | Bandwidth > 2 MHz: floor(bandwidth_MHz / 2) chunks per time segment |

---

## Grand Totals

| Metric | Value |
|--------|-------|
| Total .npy files | 232 |
| Files parsed successfully | 230 |
| Empty files (0 time segments) | 3 |
| Total time segments | 745,239 |
| **Total paper-style segments** | **78,250,915** |
| **Total duration** | **999.37 hours** (~41.6 days) |
| Ratio (paper/original) | 105x |

---

## Summary by Technology

| Tech | Files | Time Segs | Avg Chunks | Paper Segs | Duration | Avg BW (MHz) |
|------|-------|-----------|------------|------------|----------|--------------|
| DAB | 32 | 233,026 | 9.7 | 3,185,139 | 40.68 hrs | 19.8 |
| DVB-T | 34 | 178,985 | 412.1 | 68,399,314 | 873.55 hrs | 824.8 |
| FM | 41 | 108,024 | 12.0 | 1,594,226 | 20.36 hrs | 24.5 |
| GSM | 44 | 42,510 | 7.7 | 452,317 | 5.78 hrs | 16.0 |
| LTE | 41 | 137,318 | 15.9 | 2,253,438 | 28.78 hrs | 31.8 |
| TETRA | 38 | 45,376 | 5.3 | 2,366,481 | 30.22 hrs | 11.1 |

---

## Per-File Breakdown

### DAB (32 files)

| Sensor | BW (MHz) | Freq Bins | Time Segs | Chunks | Paper Segs | Duration | Filename |
|--------|----------|-----------|-----------|--------|------------|----------|----------|
| Andrew_GVA | 21 | 449 | 2,259 | 10 | 22,590 | 17.3 min | SpectrumBands_207_228_dab_Swis_207_228.npy |
| BZ_Italy | 29 | 457 | 168,013 | 14 | 2,352,182 | 30.04 hrs | SpectrumBands_203_2320_dab_Ita_203_232.npy |
| Britof | 16 | 449 | 1,612 | 8 | 12,896 | 9.9 min | SpectrumBands_180_195_dab_slv_180_196.npy |
| CYD_EPFL_S | 18 | 454 | 1,936 | 9 | 17,424 | 13.4 min | SpectrumBands_209_227_dab_Switz_209_227.npy |
| Defcon1 | 10 | 426 | 1,076 | 5 | 5,380 | 4.1 min | SpectrumBands_219_229_dab_ne_219_229.npy |
| EDLV | 7 | 426 | 752 | 3 | 2,256 | 1.7 min | SpectrumBands_174_181_dab_ge_174_181.npy |
| Espi1 | 16 | 377 | 1,829 | 8 | 14,632 | 11.2 min | SpectrumBands_203_220_dab_ger_180_196.npy |
| Geneva | 16 | 457 | 1,721 | 8 | 13,768 | 10.6 min | SpectrumBands_210_226_dab_Swis_210_226.npy |
| Giansense | 18 | 442 | 1,936 | 9 | 17,424 | 13.4 min | SpectrumBands_213_231_dab_ita_213_231.npy |
| HuldOne | 45 | 450 | 4,841 | 22 | 106,502 | 1.36 hrs | SpectrumBands_173_218_dab_cz_173_218.npy |
| Nudelsalat_RPi | 23 | 461 | 2,472 | 11 | 27,192 | 20.8 min | SpectrumBands_176_199_dab_Ger_176_199.npy |
| Rapallo_Electrosense | 8 | 458 | 860 | 4 | 3,440 | 2.6 min | SpectrumBands_222_230_dab_Ita_222_230.npy |
| Sensorix | 16 | 369 | 1,721 | 8 | 13,768 | 10.6 min | SpectrumBands_209_225_dab_Swis_209_225.npy |
| Skap_French_Riviera | 34 | 451 | 3,657 | 17 | 62,169 | 47.6 min | SpectrumBands_194_228_dab_Esp_194_228.npy |
| Somdalsbraatan | 14 | 444 | 1,505 | 7 | 10,535 | 8.1 min | SpectrumBands_225_239_dab_no_225_239.npy |
| URJC1 | 21 | 406 | 2,258 | 10 | 22,580 | 17.3 min | SpectrumBands_208_229_dab_Esp_208_229.npy |
| Valbella | 5 | 409 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_220_225_dab_Switz_220_225.npy |
| Valbella | 8 | 407 | 860 | 4 | 3,440 | 2.6 min | SpectrumBands_225_233_dab_Switz_225_233.npy |
| Zagreb | 4 | 456 | 430 | 2 | 860 | 0.7 min | SpectrumBands_204_208_dab_Cro_204_208.npy |
| alcorcon1 | 35 | 203 | 3,747 | 17 | 63,699 | 48.8 min | SpectrumBands_195_230_dab_alcorcon_195_230.npy |
| alcorcon1 | 35 | 498 | 3,747 | 17 | 63,699 | 48.8 min | alcorcon_Feb_2_21SpectrumBands_195_230_dab_alcorcon_195_230.npy |
| alcorcon1 | 35 | 474 | 3,747 | 17 | 63,699 | 48.8 min | alcorcon_Feb_3_21SpectrumBands_195_230_dab_alcorcon_195_230.npy |
| alemino_ZRH | 41 | 463 | 4,410 | 20 | 88,200 | 1.13 hrs | SpectrumBands_187_228_dab_Swis_187_228.npy |
| bcn-L | 29 | 450 | 3,120 | 14 | 43,680 | 33.5 min | SpectrumBands_191_220_dab_Esp_191_220.npy |
| dipolkurz | 21 | 442 | 2,258 | 10 | 22,580 | 17.3 min | SpectrumBands_208_229_dab_Swis_208_229.npy |
| goeppingen_I | 20 | 326 | 2,150 | 10 | 21,500 | 16.5 min | SpectrumBands_203_223_dab_Ger_203_223.npy |
| imdea_adsb | 9 | 451 | 969 | 4 | 3,876 | 3.0 min | SpectrumBands_194_203_dab_Esp_194_203.npy |
| oha_sense1 | 30 | 451 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_200_230_dab_oh_200_230.npy |
| oha_sense1 | 30 | 451 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_200_230_dab_oh_200_230.npy |
| scalessio | 6 | 454 | 645 | 3 | 1,935 | 1.5 min | SpectrumBands_201_207_dab_scalessio_201_207.npy |
| scalessio | 6 | 444 | 645 | 3 | 1,935 | 1.5 min | SpectrumBands_201_207_dab_scalessio_201_207.npy |
| vlenders_burgdorf | 8 | 423 | 861 | 4 | 3,444 | 2.6 min | SpectrumBands_208_216_dab_sw_208_216.npy |

### DVB-T (34 files)

| Sensor | BW (MHz) | Freq Bins | Time Segs | Chunks | Paper Segs | Duration | Filename |
|--------|----------|-----------|-----------|--------|------------|----------|----------|
| Andrew_GVA | 40 | 449 | 4,302 | 20 | 86,040 | 1.10 hrs | SpectrumBands_542_582_dvbt_Swis_542_582.npy |
| BGSOFIAMLADOST01 | 46 | 450 | 4,947 | 23 | 113,781 | 1.45 hrs | SpectrumBands_515_561_dvbt_bu_515_561.npy |
| BZ_Italy | 109 | 457 | 11,722 | 54 | 632,988 | 8.08 hrs | SpectrumBands_540_649_dvbt_Ita_540_649.npy |
| Britof | 58 | 449 | 6,237 | 29 | 180,873 | 2.31 hrs | SpectrumBands_557_615_dvbt_slv_557_615.npy |
| Defcon1 | 59 | 426 | 6,345 | 29 | 184,005 | 2.35 hrs | SpectrumBands_636_695_dvbt_ne_636_695.npy |
| EDLV | 33 | 426 | 3,550 | 16 | 56,800 | 43.5 min | SpectrumBands_557_590_dvbt_ge_557_590.npy |
| Espi1 | 62 | 377 | 6,668 | 31 | 206,708 | 2.64 hrs | SpectrumBands_468_530_dvbt_ger_468_530.npy |
| Geneva | 41 | 457 | 4,409 | 20 | 88,180 | 1.13 hrs | SpectrumBands_464_505_dvbt_Swis_464_505.npy |
| Giansense | 66 | 442 | 7,098 | 33 | 234,234 | 2.99 hrs | SpectrumBands_620_686_dvbt_ita_620_686.npy |
| HuldOne | 35 | 450 | 0 | 17 | 0 | 0.0 min | SpectrumBands_484_419_dvbt_cz_484_519.npy |
| IADAM | 9 | 337 | 969 | 4 | 3,876 | 3.0 min | SpectrumBands_637_646_dvbt_czk_637_646.npy |
| NFM-Electrosense-01 | 43 | 445 | 4,624 | 21 | 97,104 | 1.24 hrs | SpectrumBands_518_561_dvbt_Jap_518_561.npy |
| Nudelsalat_RPi | 76 | 461 | 8,173 | 38 | 310,574 | 3.97 hrs | SpectrumBands_580_656_dvbt_Ger_580_656.npy |
| PiSDR1 | 57 | 462 | 6,131 | 28 | 171,668 | 2.19 hrs | SpectrumBands_573_630_dvbt_po_573_630.npy |
| Rapallo_Electrosense | 103 | 458 | 11,076 | 51 | 564,876 | 7.21 hrs | SpectrumBands_468_571_dvbt_Ita_468_571.npy |
| SP8AKM | 3 | 356 | 324 | 1 | 324 | 0.2 min | SpectrumBands_667_670_dvbt_Pol_667_670.npy |
| Sashatka | 15 | 450 | 1,614 | 7 | 11,298 | 8.7 min | SpectrumBands_775_790_dvbt_rus_775_790.npy |
| Skap_French_Riviera | 62 | 451 | 6,667 | 31 | 206,677 | 2.64 hrs | SpectrumBands_473_535_dvbt_Esp_473_535.npy |
| Somdalsbraatan | 82 | 444 | 8,818 | 41 | 361,538 | 4.62 hrs | SpectrumBands_524_606_dvbt_no_524_606.npy |
| URJC1 | 80 | 406 | 8,604 | 40 | 344,160 | 4.40 hrs | SpectrumBands_475_555_dvbt_Esp_475_555.npy |
| Zagreb | 63 | 456 | 6,776 | 31 | 210,056 | 2.68 hrs | SpectrumBands_475_538_dvbt_Cro_475_538.npy |
| alcorcon1 | 5305 | 243 | 4,763 | 2,652 | 12,631,476 | 161.32 hrs | SpectrumBands_540_584_dvbt_alcorcon_540_5845.npy |
| alcorcon1 | 5305 | 495 | 4,763 | 2,652 | 12,631,476 | 161.32 hrs | alcorcon_Feb_2_21SpectrumBands_540_584_dvbt_alcorcon_540_5845.npy |
| alcorcon1 | 5305 | 471 | 4,763 | 2,652 | 12,631,476 | 161.32 hrs | alcorcon_Feb_3_21SpectrumBands_540_584_dvbt_alcorcon_540_5845.npy |
| bcn-L | 99 | 450 | 6,344 | 49 | 310,856 | 3.97 hrs | SpectrumBands_484_543_dvbt_Esp_484_583.npy |
| imdea_adsb | 10 | 451 | 1,076 | 5 | 5,380 | 4.1 min | SpectrumBands_620_630_dvbt_Esp_620_630.npy |
| leganes_rack_3 | 5305 | 449 | 4,786 | 2,652 | 12,692,472 | 162.10 hrs | SpectrumBands_540_584_dvbt_rack3_540_5845.npy |
| leganes_rack_3 | 5305 | 445 | 4,786 | 2,652 | 12,692,472 | 162.10 hrs | SpectrumBands_540_584_dvbt_rack3_540_5845.npy |
| miguel_murcia | 8 | 455 | 861 | 4 | 3,444 | 2.6 min | SpectrumBands_636_644_dvbt_Esp_636_644.npy |
| oha_sense1 | 59 | 451 | 6,344 | 29 | 183,976 | 2.35 hrs | SpectrumBands_468_527_dvbt_oh_468_527.npy |
| oha_sense1 | 59 | 446 | 6,344 | 29 | 183,976 | 2.35 hrs | SpectrumBands_468_527_dvbt_oh_468_527.npy |
| scalessio | 41 | 454 | 4,324 | 20 | 86,480 | 1.10 hrs | SpectrumBands_509_550_dvbt_scalessio_509_550.npy |
| scalessio | 41 | 444 | 4,324 | 20 | 86,480 | 1.10 hrs | SpectrumBands_509_550_dvbt_scalessio_509_550.npy |
| skyhubfr | 60 | 434 | 6,453 | 30 | 193,590 | 2.47 hrs | SpectrumBands_560_620_dvbt_fr_560_620.npy |

### FM (41 files)

| Sensor | BW (MHz) | Freq Bins | Time Segs | Chunks | Paper Segs | Duration | Filename |
|--------|----------|-----------|-----------|--------|------------|----------|----------|
| Andrew_GVA | 30 | 449 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_80_110_fm_Swis_80_110.npy |
| BGSOFIAMLADOST01 | 6 | 450 | 645 | 3 | 1,935 | 1.5 min | SpectrumBands_85_91_fm_bu_85_91.npy |
| BZ_Italy | 15 | 457 | 1,614 | 7 | 11,298 | 8.7 min | SpectrumBands_87_102_fm_Ita_87_102.npy |
| Bandon_Oregon_GAP1 | 24 | 448 | 2,581 | 12 | 30,972 | 23.7 min | SpectrumBands_86_110_fm_usa_86_110.npy |
| Britof | 35 | 449 | 3,765 | 17 | 64,005 | 49.0 min | SpectrumBands_93_128_fm_slv_93_128.npy |
| CYD_EPFL_S | 20 | 454 | 2,150 | 10 | 21,500 | 16.5 min | SpectrumBands_100_120_fm_Switz_100_120.npy |
| Defcon1 | 5 | 426 | 538 | 2 | 1,076 | 0.8 min | SpectrumBands_87_92_fm_ne_87_92.npy |
| EDLV | 6 | 426 | 645 | 3 | 1,935 | 1.5 min | SpectrumBands_100_106_fm_ge_100_106.npy |
| Espi1 | 15 | 377 | 1,614 | 7 | 11,298 | 8.7 min | SpectrumBands_87_102_fm_ger_87_102.npy |
| Geneva | 26 | 457 | 2,796 | 13 | 36,348 | 27.9 min | SpectrumBands_94_120_fm_Swis_94_120.npy |
| Giansense | 9 | 442 | 967 | 4 | 3,868 | 3.0 min | SpectrumBands_98_107_fm_ita_98_107.npy |
| HuldOne | 8 | 450 | 861 | 4 | 3,444 | 2.6 min | SpectrumBands_99_107_fm_cz_99_107.npy |
| IADAM | 25 | 337 | 2,689 | 12 | 32,268 | 24.7 min | SpectrumBands_85_110_fm_czk_85_110.npy |
| NFM-Electrosense-01 | 21 | 445 | 2,151 | 10 | 21,510 | 16.5 min | SpectrumBands_70_90_fm_Jap_207_228.npy |
| Nudelsalat_RPi | 35 | 461 | 3,764 | 17 | 63,988 | 49.0 min | SpectrumBands_85_120_fm_Ger_85_120.npy |
| Oreland | 24 | 457 | 2,581 | 12 | 30,972 | 23.7 min | SpectrumBands_86_110_fm_usa_86_110.npy |
| PiSDR1 | 10 | 462 | 1,075 | 5 | 5,375 | 4.1 min | SpectrumBands_100_110_fm_po_100_110.npy |
| Princeton1 | 24 | 443 | 2,581 | 12 | 30,972 | 23.7 min | SpectrumBands_86_110_fm_usa_86_110.npy |
| Rapallo_Electrosense | 40 | 458 | 4,301 | 20 | 86,020 | 1.10 hrs | SpectrumBands_80_120_fm_Ita_80_120.npy |
| SP8AKM | 40 | 356 | 4,301 | 20 | 86,020 | 1.10 hrs | SpectrumBands_80_120_fm_Pol_80_120.npy |
| Sashatka | 30 | 450 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_90_120_fm_rus_90_120.npy |
| Sensorix | 35 | 369 | 3,764 | 17 | 63,988 | 49.0 min | SpectrumBands_85_120_fm_Swis_85_120.npy |
| Skap_French_Riviera | 35 | 451 | 3,763 | 17 | 63,971 | 49.0 min | SpectrumBands_85_120_fm_Esp_85_120.npy |
| URJC1 | 35 | 406 | 3,764 | 17 | 63,988 | 49.0 min | SpectrumBands_85_120_fm_Esp_85_120.npy |
| Valbella | 20 | 407 | 2,150 | 10 | 21,500 | 16.5 min | SpectrumBands_100_120_fm_Switz_100_120.npy |
| Valbella | 15 | 409 | 1,614 | 7 | 11,298 | 8.7 min | SpectrumBands_87_102_fm_Switz_87_102.npy |
| Zagreb | 20 | 456 | 2,151 | 10 | 21,510 | 16.5 min | SpectrumBands_91_111_fm_Cro_91_111.npy |
| alcorcon1 | 35 | 203 | 3,747 | 17 | 63,699 | 48.8 min | SpectrumBands_85_120_fm_alcorcon_85_120.npy |
| alcorcon1 | 35 | 498 | 3,747 | 17 | 63,699 | 48.8 min | alcorcon_Feb_2_21SpectrumBands_85_120_fm_alcorcon_85_120.npy |
| alcorcon1 | 35 | 474 | 3,747 | 17 | 63,699 | 48.8 min | alcorcon_Feb_3_21SpectrumBands_85_120_fm_alcorcon_85_120.npy |
| alemino_ZRH | 15 | 463 | 1,614 | 7 | 11,298 | 8.7 min | SpectrumBands_87_102_fm_Swis_87_102.npy |
| bcn-L | 30 | 450 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_90_120_fm_Esp_90_120.npy |
| dipolkurz | 35 | 442 | 3,764 | 17 | 63,988 | 49.0 min | SpectrumBands_85_120_fm_Swis_85_120.npy |
| donostia-ategorrieta | 40 | 453 | 4,301 | 20 | 86,020 | 1.10 hrs | SpectrumBands_80_120_fm_Esp_80_120.npy |
| goeppingen_I | 20 | 326 | 2,150 | 10 | 21,500 | 16.5 min | SpectrumBands_100_120_fm_Ger_100_120.npy |
| imdea_adsb | 40 | 451 | 4,302 | 20 | 86,040 | 1.10 hrs | SpectrumBands_80_120_fm_Esp_80_120.npy |
| miguel_murcia | 20 | 455 | 2,150 | 10 | 21,500 | 16.5 min | SpectrumBands_85_105_fm_Esp_85_105.npy |
| scalessio | 40 | 454 | 4,301 | 20 | 86,020 | 1.10 hrs | SpectrumBands_80_120_fm_scalessio_80_120.npy |
| scalessio | 40 | 444 | 4,301 | 20 | 86,020 | 1.10 hrs | SpectrumBands_80_120_fm_scalessio_80_120.npy |
| skyhubfr | 8 | 434 | 860 | 4 | 3,440 | 2.6 min | SpectrumBands_102_110_fm_fr_102_110.npy |
| vlenders_burgdorf | 5 | 423 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_102_107_fm_sw_102_107.npy |

### GSM (44 files)

| Sensor | BW (MHz) | Freq Bins | Time Segs | Chunks | Paper Segs | Duration | Filename |
|--------|----------|-----------|-----------|--------|------------|----------|----------|
| BGSOFIAMLADOST01 | 11 | 450 | 1,182 | 5 | 5,910 | 4.5 min | SpectrumBands_914_925_gsm_bu_914_925.npy |
| BZ_Italy | 8 | 457 | 861 | 4 | 3,444 | 2.6 min | SpectrumBands_936_944_gsm_Ita_936_944.npy |
| Defcon1 | 15 | 426 | 1,612 | 7 | 11,284 | 8.6 min | SpectrumBands_920_935_gsm_ne_920_935.npy |
| EDLV | 9 | 426 | 969 | 4 | 3,876 | 3.0 min | SpectrumBands_943_952_gsm_ge_943_952.npy |
| Espi1 | 17 | 377 | 1,827 | 8 | 14,616 | 11.2 min | SpectrumBands_920_937_gsm_ger_920_937.npy |
| Giansense | 66 | 442 | 3,226 | 33 | 106,458 | 1.36 hrs | SpectrumBands_930_960_gsm_ita_620_686.npy |
| HuldOne | 5 | 450 | 539 | 2 | 1,078 | 0.8 min | SpectrumBands_925_930_gsm_cz_925_930.npy |
| IADAM | 5 | 337 | 539 | 2 | 1,078 | 0.8 min | SpectrumBands_921_926_gsm_czk_921_926.npy |
| NFM-Electrosense-01 | 7 | 445 | 74 | 3 | 222 | 0.2 min | SpectrumBands_85_85_gsm_Jap_850_857.npy |
| Nudelsalat_RPi | 35 | 461 | 3,765 | 17 | 64,005 | 49.0 min | SpectrumBands_925_960_gsm_Ger_925_960.npy |
| PiSDR1 | 8 | 462 | 860 | 4 | 3,440 | 2.6 min | SpectrumBands_935_943_gsm_po_935_943.npy |
| Rapallo_Electrosense | 3 | 458 | 1,397 | 1 | 1,397 | 1.1 min | SpectrumBands_934_947_gsm_Ita_934_937.npy |
| SP8AKM | 8 | 356 | 860 | 4 | 3,440 | 2.6 min | SpectrumBands_935_943_gsm_Pol_935_943.npy |
| Sashatka | 28 | 450 | 3,011 | 14 | 42,154 | 32.3 min | SpectrumBands_933_961_gsm_rus_933_961.npy |
| Skap_French_Riviera | 5 | 451 | 539 | 2 | 1,078 | 0.8 min | SpectrumBands_933_938_gsm_Esp_933_938.npy |
| Somdalsbraatan | 7 | 444 | 751 | 3 | 2,253 | 1.7 min | SpectrumBands_942_949_gsm_no_942_949.npy |
| URJC1 | 6 | 406 | 645 | 3 | 1,935 | 1.5 min | SpectrumBands_954_960_gsm_Esp_954_960.npy |
| Valbella | 8 | 409 | 861 | 4 | 3,444 | 2.6 min | SpectrumBands_936_944_gsm_Switz_936_944.npy |
| Zagreb | 11 | 456 | 1,184 | 5 | 5,920 | 4.5 min | SpectrumBands_949_960_gsm_Cro_949_960.npy |
| alcorcon1 | 5 | 243 | 535 | 2 | 1,070 | 0.8 min | SpectrumBands_924_929_gsm_alcorcon_924_929.npy |
| alcorcon1 | 4 | 243 | 429 | 2 | 858 | 0.7 min | SpectrumBands_935_939_gsm_alcorcon_935_939.npy |
| alcorcon1 | 60 | 243 | 642 | 30 | 19,260 | 14.8 min | SpectrumBands_954_960_gsm_alcorcon_9544_9604.npy |
| alcorcon1 | 5 | 495 | 535 | 2 | 1,070 | 0.8 min | alcorcon_Feb_2_21SpectrumBands_924_929_gsm_alcorcon_924_929.npy |
| alcorcon1 | 4 | 495 | 429 | 2 | 858 | 0.7 min | alcorcon_Feb_2_21SpectrumBands_935_939_gsm_alcorcon_935_939.npy |
| alcorcon1 | 60 | 495 | 642 | 30 | 19,260 | 14.8 min | alcorcon_Feb_2_21SpectrumBands_954_960_gsm_alcorcon_9544_9604.npy |
| alcorcon1 | 5 | 471 | 535 | 2 | 1,070 | 0.8 min | alcorcon_Feb_3_21SpectrumBands_924_929_gsm_alcorcon_924_929.npy |
| alcorcon1 | 4 | 471 | 429 | 2 | 858 | 0.7 min | alcorcon_Feb_3_21SpectrumBands_935_939_gsm_alcorcon_935_939.npy |
| alcorcon1 | 60 | 471 | 642 | 30 | 19,260 | 14.8 min | alcorcon_Feb_3_21SpectrumBands_954_960_gsm_alcorcon_9544_9604.npy |
| alemino_ZRH | 8 | 463 | 861 | 4 | 3,444 | 2.6 min | SpectrumBands_936_944_gsm_Swis_936_944.npy |
| bcn-L | 5 | 450 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_924_929_gsm_Esp_924_929.npy |
| dipolkurz | 3 | 442 | 321 | 1 | 321 | 0.2 min | SpectrumBands_922_925_gsm_Swis_922_925.npy |
| goeppingen_I | 23 | 326 | 2,475 | 11 | 27,225 | 20.9 min | SpectrumBands_925_948_gsm_Ger_925_948.npy |
| imdea_adsb | 5 | 451 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_924_929_gsm_Esp_924_929.npy |
| leganes_rack_3 | 5 | 449 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_924_929_gsm_rack3_924_929.npy |
| leganes_rack_3 | 4 | 449 | 430 | 2 | 860 | 0.7 min | SpectrumBands_935_939_gsm_rack3_935_939.npy |
| leganes_rack_3 | 60 | 449 | 645 | 30 | 19,350 | 14.8 min | SpectrumBands_954_960_gsm_rack3_9544_9604.npy |
| leganes_rack_3 | 5 | 445 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_924_929_gsm_rack3_924_929.npy |
| leganes_rack_3 | 4 | 445 | 430 | 2 | 860 | 0.7 min | SpectrumBands_935_939_gsm_rack3_935_939.npy |
| leganes_rack_3 | 60 | 445 | 645 | 30 | 19,350 | 14.8 min | SpectrumBands_954_960_gsm_rack3_9544_9604.npy |
| miguel_murcia | 5 | 455 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_924_929_gsm_Esp_924_929.npy |
| scalessio | 25 | 150 | 2,239 | 12 | 26,868 | 20.6 min | SpectrumBands_935_960_gsm_test_935_960.npy |
| scalessio | 9 | 454 | 969 | 4 | 3,876 | 3.0 min | SpectrumBands_939_948_gsm_scalessio_939_948.npy |
| scalessio | 9 | 444 | 969 | 4 | 3,876 | 3.0 min | SpectrumBands_939_948_gsm_scalessio_939_948.npy |
| vlenders_burgdorf | 3 | 423 | 321 | 1 | 321 | 0.2 min | SpectrumBands_922_925_gsm_sw_922_925.npy |

### LTE (41 files)

| Sensor | BW (MHz) | Freq Bins | Time Segs | Chunks | Paper Segs | Duration | Filename |
|--------|----------|-----------|-----------|--------|------------|----------|----------|
| BZ_Italy | 30 | 457 | 3,548 | 15 | 53,220 | 40.8 min | SpectrumBands_788_821_lte_Ita_791_821.npy |
| Bandon_Oregon_GAP1 | 34 | 448 | 3,656 | 17 | 62,152 | 47.6 min | SpectrumBands_734_768_lte_usa_734_768.npy |
| Britof | 30 | 449 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_slv_791_821.npy |
| CYD_EPFL_S | 30 | 454 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Switz_791_821.npy |
| Defcon1 | 30 | 426 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_ne_791_821.npy |
| EDLV | 30 | 426 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_ge_791_821.npy |
| Espi1 | 30 | 377 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_ger_791_821.npy |
| Geneva | 30 | 457 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Swis_791_821.npy |
| Giansense | 30 | 442 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_ita_791_821.npy |
| HuldOne | 31 | 450 | 3,333 | 15 | 49,995 | 38.3 min | SpectrumBands_790_821_lte_cz_790_821.npy |
| IADAM | 30 | 337 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_czk_791_821.npy |
| NFM-Electrosense-01 | 33 | 445 | 0 | 16 | 0 | 0.0 min | SpectrumBands_770_80_lte_Jap_770_803.npy |
| Nudelsalat_RPi | 30 | 461 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Ger_791_821.npy |
| Oreland | 34 | 457 | 3,656 | 17 | 62,152 | 47.6 min | SpectrumBands_734_768_lte_usa_734_768.npy |
| PiSDR1 | 31 | 462 | 3,333 | 15 | 49,995 | 38.3 min | SpectrumBands_790_821_lte_po_790_821.npy |
| Princeton1 | 34 | 443 | 3,656 | 17 | 62,152 | 47.6 min | SpectrumBands_734_768_lte_usa_734_768.npy |
| Rapallo_Electrosense | 30 | 458 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Ita_791_821.npy |
| SP8AKM | 30 | 356 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Pol_791_821.npy |
| Sensorix | 30 | 369 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Swis_791_821.npy |
| Skap_French_Riviera | 30 | 451 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Esp_791_821.npy |
| Somdalsbraatan | 31 | 444 | 3,333 | 15 | 49,995 | 38.3 min | SpectrumBands_790_821_lte_no_790_821.npy |
| URJC1 | 30 | 406 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Esp_791_821.npy |
| Valbella | 52 | 409 | 5,592 | 26 | 145,392 | 1.86 hrs | SpectrumBands_771_823_lte_Switz_771_823.npy |
| Valbella | 33 | 409 | 3,548 | 16 | 56,768 | 43.5 min | SpectrumBands_788_821_lte_Switz_788_821.npy |
| Zagreb | 30 | 456 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Cro_791_821.npy |
| alcorcon1 | 30 | 203 | 3,212 | 15 | 48,180 | 36.9 min | SpectrumBands_791_821_lte_alcorcon_791_821.npy |
| alcorcon1 | 30 | 498 | 3,212 | 15 | 48,180 | 36.9 min | alcorcon_Feb_2_21SpectrumBands_791_821_lte_alcorcon_791_821.npy |
| alcorcon1 | 30 | 474 | 3,212 | 15 | 48,180 | 36.9 min | alcorcon_Feb_3_21SpectrumBands_791_821_lte_alcorcon_791_821.npy |
| alemino_ZRH | 30 | 463 | 3,548 | 15 | 53,220 | 40.8 min | SpectrumBands_788_821_lte_Swis_791_821.npy |
| bcn-L | 30 | 450 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Esp_791_821.npy |
| dipolkurz | 30 | 442 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Swis_791_821.npy |
| donostia-ategorrieta | 30 | 453 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Esp_791_821.npy |
| goeppingen_I | 30 | 326 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Ger_791_821.npy |
| imdea_adsb | 30 | 451 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Esp_791_821.npy |
| miguel_murcia | 30 | 455 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_Esp_791_821.npy |
| oha_sense1 | 30 | 451 | 3,194 | 15 | 47,910 | 36.7 min | SpectrumBands_791_821_lte_oh_791_821.npy |
| oha_sense1 | 30 | 446 | 3,194 | 15 | 47,910 | 36.7 min | SpectrumBands_791_821_lte_oh_791_821.npy |
| scalessio | 30 | 454 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_scalessio_791_821.npy |
| scalessio | 30 | 444 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_scalessio_791_821.npy |
| skyhubfr | 62 | 434 | 6,667 | 31 | 206,677 | 2.64 hrs | SpectrumBands_760_822_lte_fr_760_822.npy |
| vlenders_burgdorf | 30 | 423 | 3,226 | 15 | 48,390 | 37.1 min | SpectrumBands_791_821_lte_sw_791_821.npy |

### TETRA (38 files)

| Sensor | BW (MHz) | Freq Bins | Time Segs | Chunks | Paper Segs | Duration | Filename |
|--------|----------|-----------|-----------|--------|------------|----------|----------|
| Andrew_GVA | 6 | 449 | 645 | 3 | 1,935 | 1.5 min | SpectrumBands_389_395_tetra_Swis_389_395.npy |
| BGSOFIAMLADOST01 | 5 | 450 | 538 | 2 | 1,076 | 0.8 min | SpectrumBands_393_398_tetra_bu_393_398.npy |
| BZ_Italy | 5 | 457 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_390_395_tetra_Ita_390_395.npy |
| Britof | 206 | 449 | 22,154 | 103 | 2,281,862 | 29.14 hrs | SpectrumBands_189_395_tetra_slv_189_395.npy |
| CYD_EPFL_S | 11 | 454 | 1,182 | 5 | 5,910 | 4.5 min | SpectrumBands_304_315_tetra_Switz_304_315.npy |
| Defcon1 | 4 | 426 | 430 | 2 | 860 | 0.7 min | SpectrumBands_390_394_tetra_ne_390_394.npy |
| EDLV | 5 | 426 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_390_395_tetra_ge_390_395.npy |
| Espi1 | 5 | 377 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_390_395_tetra_ger_390_395.npy |
| Geneva | 4 | 457 | 430 | 2 | 860 | 0.7 min | SpectrumBands_390_394_tetra_Swis_390_394.npy |
| HuldOne | 5 | 450 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_424_429_tetra_cz_424_429.npy |
| IADAM | 4 | 337 | 430 | 2 | 860 | 0.7 min | SpectrumBands_390_394_tetra_czk_390_394.npy |
| NFM-Electrosense-01 | 5 | 445 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_270_275_tetra_Jap_270_275.npy |
| PiSDR1 | 2 | 462 | 215 | 1 | 215 | 0.2 min | SpectrumBands_425_427_tetra_po_425_427.npy |
| Sashatka | 6 | 450 | 646 | 3 | 1,938 | 1.5 min | SpectrumBands_420_426_tetra_rus_420_426.npy |
| Sensorix | 4 | 369 | 430 | 2 | 860 | 0.7 min | SpectrumBands_392_396_tetra_Swis_392_396.npy |
| Skap_French_Riviera | 6 | 451 | 645 | 3 | 1,935 | 1.5 min | SpectrumBands_389_395_tetra_Esp_389_395.npy |
| Somdalsbraatan | 5 | 444 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_390_395_tetra_no_390_395.npy |
| URJC1 | 8 | 406 | 860 | 4 | 3,440 | 2.6 min | SpectrumBands_419_427_tetra_Esp_419_427.npy |
| Valbella | 6 | 409 | 645 | 3 | 1,935 | 1.5 min | SpectrumBands_389_395_tetra_Switz_389_395.npy |
| Valbella | 10 | 407 | 1,075 | 5 | 5,375 | 4.1 min | SpectrumBands_390_400_tetra_Switz_390_400.npy |
| Zagreb | 10 | 456 | 1,075 | 5 | 5,375 | 4.1 min | SpectrumBands_389_399_tetra_Cro_389_399.npy |
| alcorcon1 | 5 | 243 | 536 | 2 | 1,072 | 0.8 min | SpectrumBands_421_426_tetra_alcorcon_421_426.npy |
| alcorcon1 | 5 | 495 | 536 | 2 | 1,072 | 0.8 min | alcorcon_Feb_2_21SpectrumBands_421_426_tetra_alcorcon_421_426.npy |
| alcorcon1 | 5 | 471 | 536 | 2 | 1,072 | 0.8 min | alcorcon_Feb_3_21SpectrumBands_421_426_tetra_alcorcon_421_426.npy |
| alemino_ZRH | 5 | 463 | 537 | 2 | 1,074 | 0.8 min | SpectrumBands_390_395_tetra_Swis_390_395.npy |
| bcn-L | 5 | 450 | 538 | 2 | 1,076 | 0.8 min | SpectrumBands_419_424_tetra_Esp_419_424.npy |
| dipolkurz | 2 | 442 | 215 | 1 | 215 | 0.2 min | SpectrumBands_393_395_tetra_Swis_393_395.npy |
| donostia-ategorrieta | 8 | 453 | 860 | 4 | 3,440 | 2.6 min | SpectrumBands_419_427_tetra_Esp_419_427.npy |
| goeppingen_I | 23 | 326 | 2,474 | 11 | 27,214 | 20.9 min | SpectrumBands_371_394_tetra_Ger_371_394.npy |
| imdea_adsb | 3 | 451 | 323 | 1 | 323 | 0.2 min | SpectrumBands_392_395_tetra_Esp_392_395.npy |
| leganes_rack_3 | 5 | 449 | 539 | 2 | 1,078 | 0.8 min | SpectrumBands_421_426_tetra_rack3_421_426.npy |
| leganes_rack_3 | 5 | 445 | 539 | 2 | 1,078 | 0.8 min | SpectrumBands_421_426_tetra_rack3_421_426.npy |
| miguel_murcia | 5 | 455 | 538 | 2 | 1,076 | 0.8 min | SpectrumBands_389_394_tetra_Esp_389_394.npy |
| oha_sense1 | 6 | 451 | 645 | 3 | 1,935 | 1.5 min | SpectrumBands_389_395_tetra_oh_389_395.npy |
| scalessio | 5 | 454 | 539 | 2 | 1,078 | 0.8 min | SpectrumBands_421_426_tetra_scalessio_421_426.npy |
| scalessio | 5 | 444 | 539 | 2 | 1,078 | 0.8 min | SpectrumBands_421_426_tetra_scalessio_421_426.npy |
| skyhubfr | 4 | 434 | 430 | 2 | 860 | 0.7 min | SpectrumBands_390_394_tetra_fr_390_394.npy |
| vlenders_burgdorf | 4 | 423 | 430 | 2 | 860 | 0.7 min | SpectrumBands_390_394_tetra_sw_390_394.npy |

---

## Notes

1. **Filename typos:** Some files have parsing issues in the first frequency pair. The analysis uses the second frequency pair (after the technology label) which is more reliable.

2. **DVB-T bandwidth anomaly:** Several DVB-T files from alcorcon1 and leganes_rack_3 show 5305 MHz bandwidth due to filename parsing (the trailing digits `5845` in `540_5845` are interpreted as part of the frequency range). These files likely capture 540-584 MHz (44 MHz).

3. **Empty files (0 time segments):**
   - `SpectrumBands_9350_941_gsm_slv_935_491.npy` (GSM)
   - `SpectrumBands_484_419_dvbt_cz_484_519.npy` (DVB-T)
   - `SpectrumBands_770_80_lte_Jap_770_803.npy` (LTE)

4. **Duplicate files:** Some sensors have duplicate files with identical names (e.g., oha_sense1 has two `SpectrumBands_200_230_dab_oh_200_230.npy`).

---

## Duplicate Sequence Analysis

### Identical File Pairs (Exact Byte-for-Byte Duplicates)

| Group | Files | Sensors | Shape | Description |
|-------|-------|---------|-------|-------------|
| 1 | 3 | Britof, HuldOne, NFM-Electrosense-01 | Various (0 cols) | Empty files with identical hash |
| 2 | 2 | Defcon1, EDLV | (426, 3226) | **LTE files are IDENTICAL** |
| 3 | 2 | Skap_French_Riviera, imdea_adsb | (451, 3226) | **LTE files are IDENTICAL** |
| 4 | 2 | oha_sense1 (Sep_1, Sep_5) | (451, 3226) | **DAB files are IDENTICAL** |

**Impact:** 4 files are exact duplicates of other files. These should be deduplicated before training.

### Cross-Sensor Same-Filename Files

Many files share the same filename across different sensors (capturing the same frequency band). These are **NOT duplicates** - they contain different data from different locations:

| Filename | Copies | Sensors | Different Data? |
|----------|--------|---------|-----------------|
| `SpectrumBands_791_821_lte_Esp_791_821.npy` | 6 | Skap, URJC1, bcn-L, donostia, imdea, miguel | Yes (different shapes/means) |
| `SpectrumBands_734_768_lte_usa_734_768.npy` | 3 | Bandon_Oregon, Oreland, Princeton1 | Yes |
| `SpectrumBands_86_110_fm_usa_86_110.npy` | 3 | Bandon_Oregon, Oreland, Princeton1 | Yes |
| `SpectrumBands_791_821_lte_Swis_791_821.npy` | 3 | Geneva, Sensorix, dipolkurz | Yes |
| `SpectrumBands_924_929_gsm_Esp_924_929.npy` | 3 | bcn-L, imdea, miguel | Yes |

**Exception:** Skap_French_Riviera and imdea_adsb have **identical** LTE data (same file copied to two sensors).

### Same-Sensor Multi-Date Files

Some sensors have data from multiple dates. These are generally **different** (collected at different times):

| Sensor | Files | Same Date? | Identical? |
|--------|-------|------------|------------|
| oha_sense1 | DAB (Sep_1, Sep_5) | Different dates | **IDENTICAL** (bug) |
| oha_sense1 | DVB-T (Sep_1, Sep_5) | Different dates | Different (shape/mean differ) |
| oha_sense1 | LTE (Sep_1, Sep_5) | Different dates | Different |
| scalessio | All 6 techs (May_2, May_3) | Different dates | Different |
| leganes_rack_3 | GSM/DVB-T (May_1, May_2) | Different dates | Different |
| alcorcon1 | All techs (Feb_1, Feb_2, Feb_3) | Different dates | Different |

### Within-File Duplicate Sequences

**Result:** No duplicate columns (time segments) found within any file. Every PSD measurement in every file is unique.

### Near-Duplicate Sequences (Correlation > 0.999)

**Result:** No near-duplicate sequences found. All time segments within files are sufficiently different.

### Summary of Duplicates

| Type | Count | Impact |
|------|-------|--------|
| Exact file duplicates | 4 files | Remove before training |
| Identical cross-sensor files | 1 pair (Skap/imdea) | Remove one |
| Identical multi-date files | 1 pair (oha_sense1 DAB) | Remove one |
| **Total unique files to remove** | **5 files** | Reduces dataset from 232 to 227 unique files |

---

## Analysis Script

The analysis script used to generate this data is `analyze_dataset.py` in the project root.
