
# IACorr

## Description
This repository provides a **TreeCorr-based wrapper** for computing _Intrinsic Alignment (IA) estimators_.

The code takes as input two galaxy catalogues:

-   **positions:** traces the galaxy density field , $D$
    
-   **shapes:** traces the intrinsic shear field, $S$
    

and the random catalogues (**random_positions**, **random_shapes**) related with each sample. The code is mainly devoted to compute the estimators for light-cone catalogues, but was also adapted to be used for a box in a snapshot. The following IA estimators are currently implemented:

- 3D Auto correlation $D$ galaxy sample:

$$    \xi_{dd}(r_p,\Pi) = \frac{DD -2DR_D + R_DR_D}{R_DR_D},$$

where $r_p$ and $\Pi$ are the projected and the line-of-sight distances, respectively. $DD$, $DR_D$ and $R_DR_D$ are the normalized number of total pairs in each $r_p$ and bins according to the number of pairs in the total sample:

$$    DD(r_p,\Pi) = \frac{dd(r_p,\Pi)}{n_d(n_d-1)/2}$$

where $n_d$ is the number of objects in $D$ and $dd(r_p,\Pi)$ is the number of pairs included in the bin $(r_p,\Pi)$.

- Cross correlation between positions of $D$ and $S$ samples:

$$    \xi_{ds}(r_p,\Pi) = \frac{DD - DR_D - SR_S + R_SR_D}{R_SR_D}$$

where in this case:

$$    SD(r_p,\Pi) = \frac{sd(r_p,\Pi)}{n_sn_d - S\cap D}$$

$S\cap D$ is the number of objects that are included in both catalogues. This is computed using a cross match between the catalogues. 
- Shape-position correlation:

$$    \xi_{d+,\times}(r_p,\Pi) = \frac{S_{+,\times}D - S_{+,\times}R_D}{R_SR_D}$$

where:

$$    S_+D(r_p,\Pi) = \frac{\sum e_{+,\times}}{n_sn_d - S\cap D}$$

- The shape-position correlation in the $(r,\mu_r)$ space, $\xi_{d+}(r,\mu_r)$, where $r$ is the 3D distance and $\mu_r=\Pi/r$
- An alternative faster shape-position correlation, that assumes that the correlation between the shear field and the density field traced by a random position distribution in negligible, $S_{+,\times}R_D = 0$. Therefore:

$$    \xi_{d+,\times}(r_p,\Pi) = \frac{S_{+,\times}D}{R_SR_D}$$

- The shape-shape correlation:

$$    \xi_{+,\times}(r_p,\Pi) = \frac{S_{+,\times}S_{+,\times}}{R_SR_S}$$

- The projected correlations defined as:

$$    w_{\text{xx}}(r_p) = \int^{\Pi_\text{max}}_{-\Pi_\text{max}} \xi_\text{xx}(r_p,\Pi) d\Pi \sim \sum^{\Pi_\text{max}}_{-\Pi_\text{max}} \Delta \Pi \xi_\text{xx}(r_p,\Pi)$$

$$    w_{g+,2}(r) = \frac{5}{2} \frac{1}{4!} \int d\mu_r L^{2,2}(\mu_r) \xi(r,\mu_r) \sim \frac{5}{2} \frac{1}{4!} \sum^{1}_{-1} \Delta \mu \xi_\text{xx}(r,\mu_r)$$

- Errors and covariance matrix are computed considering $N_\text{jk}$ regions:

$$C^\text{jk}_{ij} = \frac{N_\text{jk} - 1}{N_\text{jk}}
\sum^{N_\text{jk}}_{m=1} (\xi^m_i - \bar{\xi}_i) (\xi^m_j - \bar{\xi}_j)$$

where:

 $$ \bar{\xi}_i \frac{1}{N_\text{jk}} \sum^{N_\text{jk}}_{m=1} \xi^m_i $$


## Installation

You can install **IACorr** directly from GitHub using `pip`:
```bash
pip install git+https://github.com/elizabethjg/IACorr.git
```

or:
```bash
git clone https://github.com/elizabethjg/IACorr.git
cd IACorr
pip install -e .
```

## Prerequisites
- `python`
- `nupy`
- `treecorr`
- `astropy`
- `pandas`
These will be automatically installed by `pip`

## License

This project is licensed under the MIT License. See the LICENSE file for details
