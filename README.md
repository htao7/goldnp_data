# goldnp_data

Data for the goldnp synthesis.

Any entry with data 0 or 470 is manually assigned to indicate a failed experiment.

For overall data analysis:
'all.csv' contains all data combined.
'plot.py' is made for plotting progress.

For optimization progress analysis:
'big1.csv' contains data targeting for (1) WL = 540 +- 2 absolute, (2) FWHM = 0.4 tolerance and (3) A450 maximize. R is not used for optimization.
'small1.csv' contains data targeting for (1) R = 6.5 +- 0.5 absolute, (2) FWHM = 0.4 tolerance and (3) A450 maximize. WL is not used for optimization.
'small2.csv' contains data targeting for (1) R = 3.5 +- 0.5 absolute, (2) FWHM = 0.4 tolerance and (3) A450 maximize. WL is not used for optimization.
'analyze.py' is made for SHAP and CV analysis.



