# Lévy Processes for Financial Modelling and Option Pricing

## Project Summary
This project is primarily educational and focuses on understanding the mathematical and statistical mechanisms underlying return dynamics models based on **Lévy processes**, and their applications to **option pricing**. The main objectives are:  

- Present Lévy processes, their fundamental properties, and the mathematical tools required to model financial returns.  
- Study in detail two classical Lévy-based models:  
  - **Variance–Gamma (VG) model**: obtained by gamma subordination of a Brownian motion.  
  - **Merton model**: a jump-diffusion model with Gaussian jumps.  
- Implement and compare two parameter estimation methods:  
  - Maximum Likelihood Estimation (**MLE**)  
  - Empirical Characteristic Function (**ECF**)  
- Analyze and compare the goodness of fit of the models to empirical data (using KS and Wasserstein metrics).  
- Apply the estimated models to price a **European call option** via the **Carr–Madan Fourier transform method**. 

---

## Project Structure and Links

The project is organized into **Markdown documents** and **Python scripts**:

## PDF Documentation
- [European Option Pricing with Lévy Processes](docs/European_Option_Pricing_with_Lévy_Processes.pdf) – mathematical roadmap explaining in detail each step and the intuition behind each method (mathematical demonstrations in the appendix).

## Markdown Documentation
- [Lévy Processes Overview](docs/markdown/levy-processes.md) – fundamental theory of Lévy processes.  
- [Model and Estimation](docs/markdown/option-pricing-fourier.md) – Variance–Gamma and Merton models, with parameter estimation methods (MLE, ECF).  
- [Option Pricing via Fourier](docs/markdown/option-pricing-fourier.md) – European call pricing using the Carr–Madan method.  
- [Calibration and Results](notebook_result/calibration-result.md) – parameter estimation, model comparison, and option pricing results.  

### Python Scripts
| Script                    | Purpose                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------ |
| `merton.py`               | Functions for MLE and ECF estimation of the Merton model                                               |
| `variance_gamma.py`       | Functions for ECF estimation of the Variance–Gamma model                                               |
| `levy_diagnosis.py`       | General functions to analyze log-return characteristics and implement Carr–Madan FFT pricing           |
| `vargamma_open_source.py` | External VG implementation by [D. Laptev](https://github.com/dlaptev/VarGamma/blob/master/VarGamma.py) |
| `compar_model.ipynb`      | Notebook that performs parameter estimation, goodness-of-fit tests, and option pricing                 |

---

## Quick Instructions to Run the Models

### 1. Install Dependencies

* All required Python packages are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt
```

### 3. Running the notebook

* Ensure all scripts are in the same directory as `compar_model.ipynb`. Launch the notebook:
```
jupyter notebook compar_model.ipynb
```

* Notebook steps:
  * Load log-return data and general diagnosis.
  * Estimate VG and Merton parameters via MLE and ECF.
  * Compute KS and Wasserstein distances for model fit.
  * Price European calls using Carr–Madan FFT with VG parameters.
  * Generate plots (stored in `notebook_result/figure/`).

* **Note:** FFT parameters and damping factors can be adjusted for numerical stability. Relative paths for figures are already set in Markdown and notebook files.

---
## References

- **Fiorani, F. (2004).** *Option Pricing Under the Variance Gamma Process* [Thèse de doctorat]. MPRA Paper No. 15395. Disponible en ligne: [https://mpra.ub.uni-muenchen.de/15395/1/MPRA_paper_15395.pdf](https://mpra.ub.uni-muenchen.de/15395/1/MPRA_paper_15395.pdf)

- **Carr, P., & Madan, D. B. (1999).** Option valuation using the Fast Fourier Transform. *Journal of Computational Finance*, 2(4), 61-73. PDF version disponible: [https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf](https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf)

- **Cont, R., & Tankov, P. (s.d.).** *Lecture Notes on Lévy Processes and Financial Modelling.* Disponible en ligne: [https://old.impan.pl/swiat-matematyki/notatki-z-wyklado~/tankov2.pdf](https://old.impan.pl/swiat-matematyki/notatki-z-wyklado~/tankov2.pdf)

- **Wikipédia (2025).** *Méthode de Nelder–Mead.* Disponible en ligne: [https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Nelder-Mead](https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Nelder-Mead)

- **U.S. Department of the Treasury (2025).** *Daily Treasury Bill Rates.* Disponible en ligne: [https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=2025](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_bill_rates&field_tdr_date_value=2025)

- **Laptev, D. (2016).** *VarGamma: Variance-Gamma Model Implementation.* Deposit GitHub. Disponible en ligne: [https://github.com/dlaptev/VarGamma](https://github.com/dlaptev/VarGamma)

---
## License

All Python scripts and notebooks **originally developed in this project** are released under the **MIT License**.  
External scripts, such as `vargamma_open_source.py` by [D. Laptev](https://github.com/dlaptev/VarGamma), retain their **original license** and are used here for educational purposes only.  
Please refer to the original repository for the specific licensing terms of external code.

---
## Project Directory Structure

Modeling-Jumps-in-Asset-Prices/
│
├── data/
│
├── docs/
│   ├── [European_Option_Pricing_with_Lévy_Processes.pdf](docs/European_Option_Pricing_with_Lévy_Processes.pdf)
│   └── markdown/
│       ├── [levy-processes.md](docs/markdown/levy-processes.md)
│       ├── [option-pricing-fourier.md](docs/markdown/option-pricing-fourier.md)
│       └── [model-and-estimation.md](docs/markdown/model-and-estimation.md)
│
├── requirements.txt
│
├── src/
│   ├── variance_gamma.py
│   ├── merton.py
│   ├── levy_diagnosis.py
│   └── vargamma_open_source.py
│
├── notebooks_result/
│   ├── compare_models.ipynb
│   ├── figures/
│   └── calibration-result.md
│
├── README.md
└── LICENSE

