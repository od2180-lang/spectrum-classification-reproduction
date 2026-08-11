# Electrosense PSD Spectrum Bands Dataset
This is the dataset used in the paper A. Scalingi, D. Giustiniano, R. Calvo-Palomino, N. Apostolakis, G. Bovet, "A Framework for Spectrum Classification using Crowdsensing Platforms". DOI 10.1109/INFOCOM53939.2023.10228867


The dataset contains PSD spectrum measurements collected by 47 ElectroSense Sensors. In this work the sensors are composed of a Raspberry Pi embedded board for computing and communication, and an RTL-SDR as radio front-end.

For each site, we collect 6 hours of the full spectrum scan, the RTL-SDR sweeps the full spectrum from 24 MHz to 1.7 GHz hopping over the frequencies with chunks of 2MHz of bandwidth. Then specific portion of the licence band are filtered to label the dataset.

The folders are organized per each site and we publish the spectrum portions already labeled.
