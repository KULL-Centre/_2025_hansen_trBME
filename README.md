# _2025_hansen_trBME
# A Bayesian approach to interpret time-resolved experiments using molecular simulations

This repo contains code to reproduce the analysis found in [A Bayesian approach to interpret time-resolved experiments using molecular simulations](https://doi.org/10.1101/2025.07.19.665657). Before the code can be run, (i) download the raw simulation data found on [ERDA](https://sid.erda.dk/cgi-sid/ls.py?share_id=FXhZ1bOKMA), setup the conda environment from `env.yml` and run `pip install kneefinder`. 
The file `main_paper_figs` contains the code to make the figures from the paper, but can also be adapted to run trBME code on your own. 

## Layout
Make_figures contains main_paper_figs.ipynb, which will allow you to reproduce figures from the paper. 
The following data-file directories are available in this repo:<br>
*`average_intensities`<br>
*`Intensities`<br>

The following data-file directories must be downloaded from ERDA:<br>
*`data`<br>
*`ensemble_trajs`<br>
*`metaD`<br>
*`rw_results`<br>
*`trajs`<br>

## Citation
If you use this method, please cite
*Carl G. Henning Hansen, Simone Orioli, and Kresten Lindorff-Larsen*//
A Bayesian approach to interpret time-resolved experiments using molecular simulations (2025). bioRxiv, 2025.
```
@article{henning_hansen_bayesian_2025,
	title = {A {Bayesian} approach to interpret time-resolved experiments using molecular simulations},
	url = {https://www.biorxiv.org/content/early/2025/07/22/2025.07.19.665657},
	doi = {10.1101/2025.07.19.665657},
	journal = {bioRxiv},
	author = {Henning Hansen, Carl G. and Orioli, Simone and Lindorff-Larsen, Kresten},
	year = {2025},
	note = {Publisher: Cold Spring Harbor Laboratory
\_eprint: https://www.biorxiv.org/content/early/2025/07/22/2025.07.19.665657.full.pdf},
}
```
