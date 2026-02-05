import numpy as np
import treecorr
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from scipy.special import lpmn
import math
from scipy import stats
from scipy import spatial

config_setup = dict(col_names = ['ra','dec','r_com','ep1','ep2','w'], # name of the columns of the catalogue related to the coordinate position (ra, dec and comoving distance for the lightcone, x, y and z for a box), projected ellipticity components weights in this order.
                    nbins = 10, # number of radial bins
                    rmin = 0.1, # minimum value for rp (r in case of the quadrupole)
                    rmax = 10., # maximum value for rp (r in case of the quadrupole)
                    pi_max = 60., # maximum value along l.o.s. (Pi) 
                    npi = 5, # number of bins in Pi
                    mubins = 10, # number of bins in mu
                    NPatches = 16,
                    ncores = 30, # Number of cores to run in parallel
                    slop = 0., # Resolution for treecorr
                    box = False, # Indicate if the data corresponds to a box, otherwise it will assume a lightcone
                    box_size = 150., # Indicate the box size to determine JK regions
                    grid_resolution = 10, # Controls grid r,mu resolution to compute the quadrupole. 
                    exact_position = True, # Indicates if the coordinates are exactly provided (e.g. simulated data without any error added in the position). Otherwise it will assume that the positions are not  exact and will use ra, dec for matching the catalogues. If this parameter is set as True, it will neglect box and it will set it as False
                    sky_threshold = 1.0 # Threshold for matching the catalogues in arcsecond, used if exact_position
#is set to True.
                    )

def norm_cov(cov):
    cov_norm = np.zeros_like(cov)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            cov_norm[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
    return cov_norm


class duplicate():
    """
        Finds weighted (w) number of duplicate objects in two catalogues. If exact positions (exact_positions = True) are provided, it uses x, y and z coordinates. Otherwise, it will match the objects in sky coordinates (ra,dec) within a given distance threshold in arcseconds given by threshold_arcsec. 

        Arguments:
        -----------
            cat1 (catalogue): Catalogue with object positions and weights.
            cat2 (catalogue): Catalogue with object positions and weights.
            exact_position (bool) : Indicates if the positions provided are exact (within a tolerance of 1.e-6).
            threshold_arcsec (float): threshold in arcseconds to match the catalogues using ra and dec.
        Attributes:
        -----------
            _w1 (ndarray,float): Projected correlation function across the pcor direction.
            _w2 (ndarray,float): Projected correlation function across the pcor direction for each Jackknife resampling.
            _exact_position (bool): Covariance matrix estimated using Jackknife resampling.
            _ind (ndarray,float): Index for cat1 to match cat2. 
            _mdist (ndarray,bool): Mask of the closest objects within the allowed threshold.
            _id_u (ndarray,bool): Mask of the unique neighbors.
            
    """

    def __init__(self,cat1,cat2,exact_position,threshold_arcsec=1.):

        self._exact_position = exact_position
        self._w1 = cat1.w
        self._w2 = cat2.w

        if exact_position:

            tree = spatial.cKDTree(np.array([cat1.x,cat1.y,cat1.z]).T)
            dist,ind=tree.query(np.array([cat2.x,cat2.y,cat2.z]).T)        
            self._ind,self._id_u = np.unique(ind,return_index=True)
            self._mdist = dist < 1.e-6
            
        else:

            c1 = SkyCoord(ra=np.array(cat1.ra)*u.degree, dec=np.array(cat1.dec)*u.degree)
            c2 = SkyCoord(ra=np.array(cat2.ra)*u.degree, dec=np.array(cat2.dec)*u.degree)    
            ind, sep2d, dist = c2.match_to_catalog_sky(c1)
            self._ind,self._id_u = np.unique(ind,return_index=True)
            self._mdist = np.array(sep2d.to(u.arcsec)) < threshold_arcsec

    def Nrep(self,mask1,mask2):
        
        Nw1 = self._w1[self._ind]*(mask1[self._ind]).astype(float)
        Nw2 = self._w2[self._id_u]*(mask2[self._id_u]).astype(float)
        threshold = (self._mdist[self._id_u]).astype(float)
        
        return np.sum(Nw1*Nw2*threshold)

class project_corr():
    """
    Computes the projected correlations from 2D correlations.

    Arguments:
    -----------
        xi_s (ndarray): Array of 2D correlation [npi, nbins].
        xi_s_jk (ndarray): Array of 2D correlation for each Jackknife resampling [NPatches, npi, nbins].
        rcor (ndarray): Array of projected radial separation bins.
        pcor (ndarray): Array of separation bins in the direction that is going to be projected (upper and lower limits).
        factor (float): Factor to be included in the integration.

    Attributes:
    -----------
        xip (ndarray): Projected correlation function across the pcor direction. Shape (nbins,)
        xip_jk (ndarray): Projected correlation function across the pcor direction for each Jackknife resampling. Shape (NPatches, nbins)
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        std_from_cov (ndarray): Standard deviation from the covariance. Shape (nbins,)
    """
    def __init__(self,xi_s,xi_s_jk,rcor,pcor,factor = 1):

        NPatches = len(xi_s_jk)

        # Compute delta_pi spacing 
        delta_pi = np.diff(pcor)[0]

        self.xip = factor * np.sum(xi_s * delta_pi, axis=0)  
        self.xip_jk = factor * np.sum(xi_s_jk * delta_pi, axis=1)  

        # Covariance
        NPatches = len(xi_s_jk)
        xip_mean = np.mean(self.xip_jk, axis=0)  # (nbins,)
        xi_diff = self.xip_jk - xip_mean[None, :]  # (NPatches, nbins)
        self.cov_jk = ((NPatches - 1) / NPatches) * np.einsum('ij,ik->jk', xi_diff, xi_diff)
        self.std_from_cov = np.sqrt(np.diagonal(self.cov_jk))




class compute_wgg(project_corr):

    """
        Computes the galaxy-galaxy correlation.
        
        Arguments:
        -----------
            config (dict): Configuration dictionary for the computation.
            dcat (treecorr.Catalog): Catalog of data points.
            rcat (treecorr.Catalog): Catalog of random points.        
        Attributes:
        -----------
            rp (ndarray): Array of projected radial separation bins.
            mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
            mean_rp (ndarray): Mean projected radial separation bins.
            Pi (ndarray): Array of l.o.s. separation bins.
            xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
            xi_jk (ndarray): 2D Correlation function for each Jackknife resampling.
            xip (ndarray): Projected correlation function across the l.o.s.
            xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
            cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
            cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
            std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """
    
    
    def __init__(self,dcat,rcat,config):

        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros((config['npi'], config['nbins']))
        DD = np.zeros((config['npi'], config['nbins']))
        RR = np.zeros((config['npi'], config['nbins']))
        DR = np.zeros((config['npi'], config['nbins']))

        xi_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
        
        dd_jk = np.zeros_like(xi_jk)
        dr_jk = np.zeros_like(xi_jk)
        rr_jk = np.zeros_like(xi_jk)

        
        # Pair normalization fractions
        Nd = dcat.sumw
        Nr = rcat.sumw
        NRpairs = (rcat.sumw*dcat.sumw)
        if config['box']:
            factor = 0.5
            NNpairs = (Nd*(Nd - 1))/2.
            RRpairs = (Nr*(Nr - 1))/2.
            
        else:
            NNpairs = (Nd*(Nd - 1))
            RRpairs = (Nr*(Nr - 1))
            factor = 1.0
        
        f0 = RRpairs/NNpairs
        f1 = RRpairs/NRpairs
    
        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])
        
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):

            dd = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            dr = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            rr = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
                
            dd.process(dcat,dcat, metric='Rperp', num_threads = config['ncores'])
            rr.process(rcat,rcat, metric='Rperp', num_threads = config['ncores'])
            dr.process(dcat,rcat, metric='Rperp', num_threads = config['ncores'])

            r[:] = np.copy(dd.rnom)
            mean_r[:] = np.copy(dd.meanr)
            mean_logr[:] = np.copy(dd.meanlogr)

            xi[p, :] = (dd.weight*factor*f0 - (2.*dr.weight)*f1 + rr.weight*factor) / (rr.weight*factor)
            DD[p, :] = dd.weight
            RR[p, :] = rr.weight
            DR[p, :] = dr.weight
            
            #Here I compute the variance
            func = lambda corrs: corrs[0].weight
            dd_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([dd], 'jackknife', func = func)
            dr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([dr], 'jackknife', func = func)
            rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func)

            dd.finalize()
            dr.finalize()
            rr.finalize()

        for i in range(config['NPatches']):
    
            swd = np.sum(dcat.w[~(dcat.patch == i)])
            swr = np.sum(rcat.w[~(rcat.patch == i)])
            NNpairs_JK = (swd*(swd - 1))
            RRpairs_JK = (swr*(swr - 1))
            
            if config['box']:
                NNpairs_JK /= 2.
                RRpairs_JK /= 2.
            
            NRpairs_JK = (swd*swr)
            
            f0_jk = RRpairs_JK/NNpairs_JK
            f1_jk = RRpairs_JK/NRpairs_JK
    
            xi_jk[i, :, :] = (dd_jk[i, :, :]*f0_jk*factor - (2.*dr_jk[i, :, :])*f1_jk + rr_jk[i, :, :]*factor) / (rr_jk[i, :, :]*factor)
    
        xi[np.isinf(xi)] = 0. #It sets to 0 the values of xi_gp that are infinite
        xi[np.isnan(xi)] = 0. #It sets to 0 the values of xi_gp that are null
    
        xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5

        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.Pi = xPi
        self.xi = xi
        self.xi_jk = xi_jk
        self.DD = DD
        self.DR = DR
        self.RR = RR
        self.Pi_bins = Pi
        
        project_corr.__init__(self,self.xi,self.xi_jk,self.rp,Pi)
        self.cov_jk_norm = norm_cov(self.cov_jk)


class compute_wgg_cross(project_corr):

    """
        Computes the cross galaxy-galaxy correlation.
        
        Arguments:
        -----------
            dcat1 (treecorr.Catalog): Catalog of data points.
            dcat2 (treecorr.Catalog): Catalog of data points.
            rcat1 (treecorr.Catalog): Catalog of random points for dcat1.        
            rcat2 (treecorr.Catalog): Catalog of random points for dcat2.        
            config (dict): Configuration dictionary for the computation.
        Attributes:
        -----------
            rp (ndarray): Array of projected radial separation bins.
            mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
            mean_rp (ndarray): Mean projected radial separation bins.
            Pi (ndarray): Array of l.o.s. separation bins.
            xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
            xi_jk (ndarray): 2D Correlation function for each Jackknife resampling.
            xip (ndarray): Projected correlation function across the l.o.s.
            xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
            cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
            cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
            std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """
    
    
    def __init__(self,dcat1,rcat1,dcat2,rcat2,config):
        # to finde duplicates
        
        dup = duplicate(dcat1,dcat2,config['exact_position'],config['sky_threshold'])
        dup_random = duplicate(rcat1,rcat2,config['exact_position'],config['sky_threshold'])

        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros((config['npi'], config['nbins']))
        xi_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
        
        d1d2_jk = np.zeros_like(xi_jk)
        d1r2_jk = np.zeros_like(xi_jk)
        d2r1_jk = np.zeros_like(xi_jk)
        r1r2_jk = np.zeros_like(xi_jk)

        # get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation   
        Nrep = dup.Nrep(np.ones(len(dcat1.w)).astype(bool),np.ones(len(dcat2.w)).astype(bool))
        Nrep_rand = dup_random.Nrep(np.ones(len(rcat1.w)).astype(bool),np.ones(len(rcat2.w)).astype(bool))

        
        # Pair normalization fractions
        N1N2pairs = (dcat1.sumw*dcat2.sumw) - Nrep
        N1R2pairs = (dcat1.sumw*rcat2.sumw)
        N2R1pairs = (dcat2.sumw*rcat1.sumw)
        R1R2pairs = (rcat2.sumw*rcat1.sumw) - Nrep_rand
    
        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])

        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):

            d1d2 = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            d1r2 = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')


            d2r1 = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            r1r2 = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
           
                
            d1d2.process(dcat1,dcat2, metric='Rperp', num_threads = config['ncores'])
            d1r2.process(dcat1,rcat2, metric='Rperp', num_threads = config['ncores'])
            d2r1.process(dcat2,rcat1, metric='Rperp', num_threads = config['ncores'])
            r1r2.process(rcat1,rcat2, metric='Rperp', num_threads = config['ncores'])
            
            r[:] = np.copy(d1d2.rnom)
            mean_r[:] = np.copy(d1d2.meanr)
            mean_logr[:] = np.copy(d1d2.meanlogr)

            xi[p, :] = (d1d2.weight/N1N2pairs - d1r2.weight/N1R2pairs - d2r1.weight/N2R1pairs + r1r2.weight/R1R2pairs) / (r1r2.weight/R1R2pairs)

            #Here I compute the variance
            func = lambda corrs: corrs[0].weight
            d1d2_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([d1d2], 'jackknife', func = func)
            d1r2_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([d1r2], 'jackknife', func = func)
            d2r1_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([d2r1], 'jackknife', func = func)
            r1r2_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([r1r2], 'jackknife', func = func)

            d1d2.finalize()
            d1r2.finalize()
            d2r1.finalize()
            r1r2.finalize()

        for i in range(config['NPatches']):
    
            swd1 = np.sum(dcat1.w[~(dcat1.patch == i)])
            swd2 = np.sum(dcat2.w[~(dcat2.patch == i)])
            swr1 = np.sum(rcat1.w[~(rcat1.patch == i)])
            swr2 = np.sum(rcat2.w[~(rcat2.patch == i)])

            Nd1d2_JK = swd1*swd2 - dup.Nrep(dcat1.patch != i,dcat2.patch != i)
            Nr1r2_JK = swr1*swr2 - dup_random.Nrep(rcat1.patch != i,rcat2.patch != i)
            N1R2_JK = (swd1*swr2)
            N2R1_JK = (swd2*swr1)
            
            xi_jk[i, :, :] = (d1d2_jk[i, :, :]/Nd1d2_JK - d1r2_jk[i, :, :]/(swd1*swr2) - d2r1_jk[i, :, :]/(swd2*swr1) + r1r2_jk[i, :, :]/(swr1*swr2)) / (r1r2_jk[i, :, :]/Nr1r2_JK)
    
        xi[np.isinf(xi)] = 0. #It sets to 0 the values of xi_gp that are infinite
        xi[np.isnan(xi)] = 0. #It sets to 0 the values of xi_gp that are null
    
        xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5

        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.Pi = xPi
        self.xi = xi
        self.xi_jk = xi_jk
        self.Pi_bins = Pi
        project_corr.__init__(self,self.xi,self.xi_jk,self.rp,Pi)
        self.cov_jk_norm = norm_cov(self.cov_jk)


class compute_wgp(project_corr):
    
    """
    Computes the galaxy-shear correlation.
    
    Arguments:
    -----------
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.
        config (dict): Configuration dictionary for the computation.
    Attributes:
    -----------
        rp (ndarray): Array of projected radial separation bins.
        mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
        mean_rp (ndarray): Mean projected radial separation bins.
        Pi (ndarray): Array of l.o.s. of the mean value pi separation bins.
        Pi_bins (ndarray): Array of l.o.s. of the upper and lower limits of the separation bins.
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
        xi_jk (ndarray): 2D Correlation function for each Jackknife resampling.
        xip (ndarray): Projected correlation function across the l.o.s.
        xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """
    
    def __init__(self,pcat,scat,rpcat,rscat,config):

        # to finde duplicates
        dup = duplicate(pcat,scat,config['exact_position'],config['sky_threshold'])
        dup_random = duplicate(rpcat,rscat,config['exact_position'],config['sky_threshold'])
        
        self.sd = []
        self.rr = []
        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros((config['npi'], config['nbins']))
        SD = np.zeros((config['npi'], config['nbins']))
        xi_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
        xi_x = np.zeros((config['npi'], config['nbins']))
        xi_x_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
      
        sr_jk = np.zeros_like(xi_jk)
        sd_jk = np.zeros_like(xi_jk)
        rr_jk = np.zeros_like(xi_jk)
        sr_x_jk = np.zeros_like(xi_jk)
        sd_x_jk = np.zeros_like(xi_jk)

        # get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation   
        Nrep = dup.Nrep(np.ones(len(pcat.w)).astype(bool),np.ones(len(scat.w)).astype(bool))
        Nrep_rand = dup_random.Nrep(np.ones(len(rpcat.w)).astype(bool),np.ones(len(rscat.w)).astype(bool))

        
        NGtot = pcat.sumw*scat.sumw - Nrep
        RRtot = rpcat.sumw*rscat.sumw - Nrep_rand
        RGtot = rpcat.sumw*scat.sumw
        f0 = RRtot / NGtot
        f1 = RRtot / RGtot

        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])

        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            rr = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            sd = treecorr.NGCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            sr = treecorr.NGCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
                
            sd.process(pcat, scat, metric='Rperp', num_threads = config['ncores'])
            sr.process(rpcat, scat, metric='Rperp', num_threads = config['ncores'])
            rr.process(rpcat, rscat, metric='Rperp', num_threads = config['ncores'])

            r[:] = np.copy(rr.rnom)
            mean_r[:] = np.copy(rr.meanr)
            mean_logr[:] = np.copy(rr.meanlogr)
    
            xi[p, :] = (f0 * (sd.xi * sd.weight) - f1 * (sr.xi * sr.weight) ) / rr.weight
            xi_x[p, :] = (f0 * (sd.xi_im * sd.weight) - f1 * (sr.xi_im * sr.weight) ) / rr.weight
            SD[p, :] = sd.xi
        
            #Here I compute the variance
            func_sd = lambda corrs: corrs[0].xi * corrs[0].weight
            func_sd_x = lambda corrs: corrs[0].xi_im * corrs[0].weight
            func_rr = lambda corrs: corrs[0].weight
            sd_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd)
            sr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = func_sd)
            rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func_rr)
            sd_x_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd_x)
            sr_x_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = func_sd_x)

            self.sd += [sd]
            self.rr += [rr]
        
            #sd.finalize()
            #sr.finalize()
            #rr.finalize()

        for i in range(config['NPatches']):
            swd  = np.sum(pcat.w[(pcat.patch != i)])
            sws  = np.sum(scat.w[(scat.patch != i)])
            swr = np.sum(rpcat.w[(rpcat.patch != i)])
            swrs = np.sum(rscat.w[(rscat.patch != i)])

            
            NGtot_JK = swd*sws - dup.Nrep(pcat.patch != i,scat.patch != i)
            RRtot_JK = swr*swrs - dup_random.Nrep(rpcat.patch != i,rscat.patch != i)
            RGtot_JK = swr*sws           
            f0_JK = RRtot_JK / NGtot_JK
            f1_JK = RRtot_JK / RGtot_JK
            xi_jk[i, :, :] = (f0_JK * sd_jk[i, :, :] - f1_JK * sr_jk[i, :, :]) / rr_jk[i, :, :]
            xi_x_jk[i, :, :] = (f0_JK * sd_x_jk[i, :, :] - f1_JK * sr_x_jk[i, :, :]) / rr_jk[i, :, :]
    
    
        xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5
        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.Pi = xPi
        self.xi = xi
        self._sd = SD
        self.xi_jk = xi_jk
        self._xi_x = xi_x
        self._xi_x_jk = xi_x_jk
        self.Pi_bins = Pi
        
        project_corr.__init__(self,self.xi,self.xi_jk,self.rp,Pi)
        self.cov_jk_norm = norm_cov(self.cov_jk)

class compute_wpp(project_corr):
    
    """
    Computes the shape-shape correlation.
    Arguments:
        config (dict): Configuration dictionary for the computation.
        scat (treecorr.Catalog): Catalog of galaxy shapes.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
    Attributes:
        rp (ndarray): Array of projected radial separation bins.
        mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
        mean_rp (ndarray): Mean projected radial separation bins.
        Pi (ndarray): Array of l.o.s. of the mean value pi separation bins.
        Pi_bins (ndarray): Array of l.o.s. of the upper and lower limits of the separation bins.
        xi (ndarray): shape-shape(++) 2-D Correlation function in bins of projected and l.o.s distance.
        xi_x (ndarray): shape-shape(--) 2-D Correlation function in bins of projected and l.o.s distance.
        xi_jk (ndarray): shape-shape(++) 2-D Correlation function for each Jackknife resampling.
        xi_x_jk (ndarray): shape-shape(--) 2-D Correlation function for each Jackknife resampling.
        xip (ndarray): Projected shape-shape(++) correlation function across the l.o.s.
        xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """
    
    def __init__(self,scat,rscat,config):
        self.ss = []
        self.rr = []
        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros((config['npi'], config['nbins']))
        xi_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
        xi_x = np.zeros((config['npi'], config['nbins']))
        xi_x_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
      
        ss_jk = np.zeros_like(xi_jk)
        rr_jk = np.zeros_like(xi_jk)
        ss_x_jk = np.zeros_like(xi_jk)

        # get pair-normalisation factors 

        Ns = scat.sumw
        Nr = rscat.sumw
        GGtot = (Ns*(Ns - 1))
        RRtot = (Nr*(Nr - 1))
        f0 = RRtot / GGtot

        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])  
        
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            ss = treecorr.GGCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            rr = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
           
            ss.process(scat,scat, metric='Rperp', num_threads = config['ncores'])
            rr.process(rscat,rscat, metric='Rperp', num_threads = config['ncores'])

            r[:] = np.copy(rr.rnom)
            mean_r[:] = np.copy(rr.meanr)
            mean_logr[:] = np.copy(rr.meanlogr)
            
            xi[p, :] = (f0 * (ss.xip * ss.weight)) / (rr.weight)
            xi_x[p, :] = (f0 * (ss.xim * ss.weight)) / (rr.weight)
        
            #Here I compute the variance
            func_ss = lambda corrs: corrs[0].xip * corrs[0].weight
            func_ss_x = lambda corrs: corrs[0].xim_im * corrs[0].weight 
            func_rr = lambda corrs: corrs[0].weight
            ss_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([ss], 'jackknife', func = func_ss)

            rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func_rr)
            ss_x_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([ss], 'jackknife', func = func_ss_x)

            self.ss += [ss]
            self.rr += [rr]
        
        for i in range(config['NPatches']):
            sws  = np.sum(scat.w[~(scat.patch == i)])
            swrs = np.sum(rscat.w[~(rscat.patch == i)])
                        
            GGtot_JK = (sws*(sws - 1))/2.
            RRtot_JK = (swrs*(swrs - 1))/2.

            f0_JK = RRtot_JK / GGtot_JK
            xi_jk[i, :, :] = (f0_JK * ss_jk[i, :, :]) / (rr_jk[i, :, :])
            xi_x_jk[i, :, :] = (f0_JK * ss_x_jk[i, :, :]) / (rr_jk[i, :, :])
    
    
        xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5
        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.Pi = xPi
        self.Pi_bins = Pi
        self.xi = 0.5*(xi+xi_x)
        self.xi_jk = 0.5*(xi_jk+xi_x_jk)
        self._xi_x = 0.5*(xi-xi_x)
        self._xi_x_jk = 0.5*(xi_jk-xi_x_jk)

        project_corr.__init__(self,self.xi,self.xi_jk,self.rp,Pi)
        self.cov_jk_norm = norm_cov(self.cov_jk)

class compute_delta_sigma():
    
    """
    Computes the galaxy-shear correlation.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
    Attributes:
    -----------
        rp (ndarray): Array of projected radial separation bins.
        mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
        mean_rp (ndarray): Mean projected radial separation bins.
        Pi (ndarray): Array of l.o.s. of the mean value pi separation bins.
        Pi_bins (ndarray): Array of l.o.s. of the upper and lower limits of the separation bins.
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
        xi_jk (ndarray): 2D Correlation function for each Jackknife resampling.
        xip (ndarray): Projected correlation function across the l.o.s.
        xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """
    
    def __init__(self,pcat,scat,config):

        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros(config['nbins'])
        xi_jk = np.zeros((config['NPatches'], config['nbins']))
        xi_x = np.zeros(config['nbins'])
        xi_x_jk = np.zeros((config['NPatches'], config['nbins']))
      

        sd_jk = np.zeros_like(xi_jk)


        # get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation   

        # now loop over Pi bins, and compute w(r_p | Pi)
            
        sd = treecorr.NGCorrelation(nbins=config['nbins'], 
                                    min_sep=config['rmin'], 
                                    max_sep=config['rmax'], 
                                    bin_slop=config['slop'], brute = False, 
                                    verbose=0, var_method = 'jackknife')
            
        sd.process(pcat, scat, metric='Rperp', num_threads = config['ncores'])

        r[:] = np.copy(sd.rnom)
        mean_r[:] = np.copy(sd.meanr)
        mean_logr[:] = np.copy(sd.meanlogr)

        
        #Here I compute the variance
        func_sd = lambda corrs: corrs[0].xi
        func_sd_x = lambda corrs: corrs[0].xi_im
        sd_jk, weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd)
        sd_x_jk, weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd_x)


        for i in range(config['NPatches']):
            xi_jk[i, :] = sd_jk[i, :]
            xi_x_jk[i, :] = sd_x_jk[i, :]
    
    
        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.xi = sd.xi
        self.xi_jk = xi_jk
        self._xi_x = sd.xi_im
        self._xi_x_jk = xi_x_jk


class compute_fast_wgp(project_corr):
    
    """
    Allows to compute the projected galaxy-shear correlation (wg+). This version allows the pre-computation of the random number of pairs and then it can be executed to compute the galaxy-shear correlation varying the shapes of the galaxies in the shape catalogue.
    
    Arguments:
    -----------
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
        config (dict): Configuration dictionary for the computation.
    Methods:
    -----------
        execute(new_shapes=False,shapes=[]): Allows the computation of the projected galaxy-shape correlation. It can be used giving a different shape catalogue as input.
    """
    
    def __init__(self,pcat,scat,rpcat,rscat,config):

        # to finde duplicates
        dup = duplicate(pcat,scat,config['exact_position'],config['sky_threshold'])
        dup_random = duplicate(rpcat,rscat,config['exact_position'],config['sky_threshold'])
        
        self.config = config
        self.pcat = pcat
        self.rpcat = rpcat
        self.rscat = rscat
        self.dup = dup
        self.dup_random = dup_random
        self.scat = scat
        self.rr_p = []
        
        # arrays to store the output
         
        self.rr_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])
        
        print('Computing normalisation factors and the number of pairs in the random catalogues...')
        # get pair-normalisation factors = total sum of (non-duplicate) weighted pairs with unlimited separation   
        Nrep = self.dup.Nrep(np.ones(len(self.pcat.w)).astype(bool),np.ones(len(scat.w)).astype(bool))
        Nrep_rand = self.dup_random.Nrep(np.ones(len(self.rpcat.w)).astype(bool),np.ones(len(self.rscat.w)).astype(bool))
        
        NGtot = self.pcat.sumw*scat.sumw - Nrep
        RRtot = self.rpcat.sumw*self.rscat.sumw - Nrep_rand

        self.f0 = RRtot / NGtot
        f0_JK = np.zeros(config['NPatches'])
        
        for i in range(config['NPatches']):
            swd  = np.sum(pcat.w[(pcat.patch != i)])
            sws  = np.sum(scat.w[(scat.patch != i)])
            swr = np.sum(rpcat.w[(self.rpcat.patch != i)])
            swrs = np.sum(rscat.w[(self.rscat.patch != i)])

            NGtot_JK = swd*sws - dup.Nrep(self.pcat.patch != i,scat.patch != i)
            RRtot_JK = swr*swrs - dup_random.Nrep(self.rpcat.patch != i,self.rscat.patch != i)
            f0_JK[i] = RRtot_JK / NGtot_JK

        self.f0_JK = np.repeat(f0_JK,config['nbins']*config['npi']).reshape(config['NPatches'], config['npi'], config['nbins'])
        
        
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            rr = treecorr.NNCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            rr.process(rpcat, rscat, metric='Rperp', num_threads = config['ncores'])
            self.rr_p += [rr]

            func_rr = lambda corrs: corrs[0].weight
            self.rr_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = func_rr)
        print('Everything ok :) Finished')
        
    def use_new_shapes(self,shapes):
        """
        Define new shape catalogue to compute the galaxy-shear correlation.

        Arguments
        ----------
        shapes : list
            Hosts the catalogue of the shapes (pandas array).
        Attributes:
        ----------
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        """
        
        config = self.config
        
        del(self.scat)
        if config['box']:
            x, y, z, g1, g2, w = config['col_names']
            zshift = 1.e6 #to put the observer far away                   
            scat  = treecorr.Catalog(g1=shapes[g1], 
                                     g2 = shapes[g2],
                                     x = shapes[x], 
                                     y = shapes[z]+zshift,  
                                     z = shapes[y], 
                                     w = shapes[w], 
                                     patch_centers = self.pcat.patch_centers)
        
        else:   
            ra, dec, r, g1, g2, w = config['col_names']
            scat  = treecorr.Catalog(g1 = shapes[g1],
                                          g2 = shapes[g2],
                                          ra=shapes[ra], 
                                          dec=shapes[dec], 
                                          r=shapes[r], 
                                          w = shapes[w], 
                                          patch_centers = self.pcat.patch_centers, 
                                          ra_units='deg', dec_units='deg')
            
        self.scat = scat
    
    def execute(self):
        """
        Compute the galaxy-shear correlation.

        Attributes:
        ----------
        rp (ndarray): Array of projected radial separation bins.
        mean_logrp (ndarray): Mean logarithmic projected radial separation bins.
        mean_rp (ndarray): Mean projected radial separation bins.
        Pi (ndarray): Array of l.o.s. of the mean value pi separation bins.
        Pi_bins (ndarray): Array of l.o.s. of the upper and lower limits of the separation bins.        
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
        xi_jk (ndarray): 2D Correlation function for each Jackknife resampling.
        xip (ndarray): Projected correlation function across the l.o.s.
        xip_jk (ndarray): Projected correlation function across the l.o.s for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
        """
        
        config = self.config
        scat = self.scat

        # arrays to store the output
        r     = np.zeros(config['nbins'])
        mean_r     = np.zeros(config['nbins'])
        mean_logr     = np.zeros(config['nbins'])
        xi = np.zeros((config['npi'], config['nbins']))
        xi_jk = np.zeros((config['NPatches'], config['npi'], config['nbins']))
      
        sr_jk = np.zeros_like(xi_jk)
        sd_jk = np.zeros_like(xi_jk)


        Pi = np.linspace(-1.*config['pi_max'], config['pi_max'], config['npi']+1)
        pibins = zip(Pi[:-1],Pi[1:])

        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            rr = self.rr_p[p]
            
            sd = treecorr.NGCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
            
            sr = treecorr.NGCorrelation(nbins=config['nbins'], 
                                        min_sep=config['rmin'], 
                                        max_sep=config['rmax'], 
                                        min_rpar=plow, max_rpar=phigh,
                                        bin_slop=config['slop'], brute = False, 
                                        verbose=0, var_method = 'jackknife')
                
            sd.process(self.pcat, scat, metric='Rperp', num_threads = config['ncores'])

            r[:] = np.copy(rr.rnom)
            mean_r[:] = np.copy(rr.meanr)
            mean_logr[:] = np.copy(rr.meanlogr)
    
            xi[p, :] = (self.f0 * (sd.xi * sd.weight) ) / rr.weight
        
            #Here I compute the variance
            func_sd = lambda corrs: corrs[0].xi * corrs[0].weight
            func_sd_x = lambda corrs: corrs[0].xi_im * corrs[0].weight
            func_rr = lambda corrs: corrs[0].weight
            sd_jk[:, p, :], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = func_sd)
        
            #sd.finalize()
            #sr.finalize()
            #rr.finalize()

        xi_jk = (self.f0_JK * sd_jk) / self.rr_jk
    
        xPi=(Pi[:-1]+Pi[1:])/2 #It returns an array going from -9.5,-8.5,...,8.5,9.5
        self.rp = r
        self.mean_rp = mean_r
        self.mean_logrp = mean_logr
        self.Pi = xPi
        self.xi = xi
        self.xi_jk = xi_jk

        project_corr.__init__(self,self.xi,self.xi_jk,self.rp,Pi)
        self.cov_jk_norm = norm_cov(self.cov_jk)

class compute_wgp2(project_corr):
    """
    Computes the quadrupole component for galaxy-shear correlation. Eq 7 in arXiv 2307.02545.
    
    Arguments:
    -----------
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
        config (dict): Configuration dictionary for the computation.
    Attributes:
    -----------
        r (ndarray): Array of projected radial separation bins.
        mu (ndarray): Array of mu (cosine of the angle) separation bins.
        xi (ndarray): 2D galaxy-shear plus correlation function in bins of 3D radial coordinate and mu.
        xi_jk (ndarray): 2D galaxy-shear plus correlation function for each Jackknife resampling.
        xip (ndarray): Projected correlation function across mu.
        xip_jk (ndarray): Projected correlation function across mu for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """

    
    def __init__(self,pcat, scat, rpcat, rscat,config):

        # to finde duplicates
        dup = duplicate(pcat,scat,config['exact_position'],config['sky_threshold'])
        dup_random = duplicate(rpcat,rscat,config['exact_position'],config['sky_threshold'])

        nbins = config['nbins']*config['grid_resolution']
        pi_max = config['rmax']-0.1
        npi = config['mubins']*config['grid_resolution']
        
        # get pair-normalisation factors

        Nrep = dup.Nrep(np.ones(len(pcat.w)).astype(bool),np.ones(len(scat.w)).astype(bool))
        Nrep_rand = dup_random.Nrep(np.ones(len(rpcat.w)).astype(bool),np.ones(len(rscat.w)).astype(bool))

        
        NGtot = scat.sumw*pcat.sumw - Nrep
        RRtot = rpcat.sumw*rscat.sumw - Nrep_rand
        RGtot = rpcat.sumw*scat.sumw
        f0 = RRtot / NGtot
        f1 = RRtot / RGtot

        # Pi bins
        Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
        r_bins  = np.logspace(np.log10(config['rmin']),np.log10(config['rmax']), nbins+1)

        # Arrays to save results
        gamma_sd = np.array([])
        gamma_sr = np.array([])
        gamma_x_sd = np.array([])
        gamma_x_sr = np.array([])
        
        npairs_sd = np.array([])
        npairs_sr = np.array([])
        npairs_rr = np.array([])
        
        r = np.array([])
        pi = np.array([])
        
        gamma_sd_jk  = np.zeros((config['NPatches'], npi*nbins))
        gamma_sr_jk  = np.zeros((config['NPatches'], npi*nbins))
        gamma_x_sd_jk  = np.zeros((config['NPatches'], npi*nbins))
        gamma_x_sr_jk  = np.zeros((config['NPatches'], npi*nbins))
        
        npairs_sd_jk = np.zeros((config['NPatches'], npi*nbins))
        npairs_sr_jk = np.zeros((config['NPatches'], npi*nbins))
        npairs_rr_jk = np.zeros((config['NPatches'], npi*nbins))
        
        f0_jk = np.zeros(config['NPatches'])
        f1_jk = np.zeros(config['NPatches'])
    
        xPi=(Pi[:-1]+Pi[1:])*0.5
        pibins = zip(Pi[:-1],Pi[1:])
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            
            r0 = max(config['rmin'],np.abs(plow))
            nbins0 = sum(r_bins >= r0)
            
            rr = treecorr.NNCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife')
            sd = treecorr.NGCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife')
            sr = treecorr.NGCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife')  
            
            sd.process(pcat, scat, metric='Euclidean', num_threads = config['ncores'])
            gamma_sd = np.append(gamma_sd,sd.xi)
            gamma_x_sd = np.append(gamma_x_sd,sd.xi_im)
            npairs_sd = np.append(npairs_sd,sd.weight)
            
            sr.process(rpcat, scat, metric='Euclidean', num_threads = config['ncores'])
            gamma_sr = np.append(gamma_sr,sr.xi)
            gamma_x_sr = np.append(gamma_x_sr,sr.xi_im)
            npairs_sr = np.append(npairs_sr,sr.weight)
            
            rr.process(rpcat, rscat, metric='Euclidean', num_threads = config['ncores'])
            npairs_rr = np.append(npairs_rr,rr.weight)
    
    
            #Here I compute the variance
            f_pairs   = lambda corrs: corrs[0].weight
            f_gamma   = lambda corrs: corrs[0].xi
            f_gamma_x = lambda corrs: corrs[0].xi_im
            gamma_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_gamma)
            gamma_sr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = f_gamma)
            gamma_x_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_gamma_x)
            gamma_x_sr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = f_gamma_x)
            
            npairs_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_pairs)
            npairs_sr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sr], 'jackknife', func = f_pairs)
            npairs_rr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = f_pairs)
    
            r  = np.append(rr.rnom,r)
            pi = np.append(xPi[p]*np.ones(len(rr.rnom)),pi)
        
        for i in range(config['NPatches']):
            ppairs_jk = np.sum(pcat.w[pcat.patch != i])
            spairs_jk = np.sum(scat.w[scat.patch != i])
            rppairs_jk = np.sum(rpcat.w[rpcat.patch != i])        
            rspairs_jk = np.sum(rscat.w[rscat.patch != i])        
            
            NGtot_JK = ppairs_jk*spairs_jk - dup.Nrep(pcat.patch != i,scat.patch != i)
            RRtot_JK = rppairs_jk*rspairs_jk - dup_random.Nrep(rpcat.patch != i,rscat.patch != i) 
            RGtot_JK = spairs_jk*rppairs_jk 
    
            f0_jk[i] = RRtot_JK / NGtot_JK
            f1_jk[i] = RRtot_JK / RGtot_JK
        
        f0_jk = np.repeat(f0_jk,config['nbins']*config['mubins']).reshape(config['NPatches'],config['mubins'],config['nbins'])
        f1_jk = np.repeat(f1_jk,config['nbins']*config['mubins']).reshape(config['NPatches'],config['mubins'],config['nbins'])
        
        gamma_sd_jk  = gamma_sd_jk[:,:len(r)]
        gamma_sr_jk = gamma_sr_jk[:,:len(r)]
        gamma_x_sd_jk = gamma_x_sd_jk[:,:len(r)]
        gamma_x_sr_jk = gamma_x_sr_jk[:,:len(r)]
        
        npairs_sd_jk = npairs_sd_jk[:,:len(r)]
        npairs_sr_jk = npairs_sr_jk[:,:len(r)]
        npairs_rr_jk = npairs_rr_jk[:,:len(r)]

        # Now bin over r and mu
        m = np.abs(pi/r) < 1.
        
        self._pi = pi
        self._r  = r
        self._gamma_sd  = gamma_sd
        self._gamma_sr  = gamma_sr
        self._npairs_sd  = npairs_sd
        self._npairs_sr  = npairs_sr
        self._npairs_rr  = npairs_rr
        
        all_sums = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                               [gamma_sd[m]*npairs_sd[m],
                                                gamma_sr[m]*npairs_sr[m],
                                                gamma_x_sd[m]*npairs_sd[m],
                                                gamma_x_sr[m]*npairs_sr[m],
                                                npairs_rr[m]
                                               ], 
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
        
        sum_gamma_sd_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                (gamma_sd_jk*npairs_sd_jk)[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )

        sum_gamma_sr_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                (gamma_sr_jk*npairs_sr_jk)[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
        
        sum_gamma_x_sd_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                (gamma_x_sd_jk*npairs_sd_jk)[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )

        sum_gamma_x_sr_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                (gamma_x_sr_jk*npairs_sr_jk)[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
        
        sum_pairs_rr_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                npairs_rr_jk[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )

        # Compute the xi and JK
        mu = all_sums.x_edge[:-1] + 0.5*np.diff(all_sums.x_edge)
        R = 10**(all_sums.y_edge[:-1] + 0.5*np.diff(all_sums.y_edge))
        sd, sr, sxd, sxr, rr = all_sums.statistic
        
        sd_jk = sum_gamma_sd_jk.statistic
        sr_jk = sum_gamma_sr_jk.statistic
        sxd_jk = sum_gamma_x_sd_jk.statistic
        sxr_jk = sum_gamma_x_sr_jk.statistic
        rr_jk = sum_pairs_rr_jk.statistic

        self.sd = sd
        self.sr = sr
        self.rr = rr
        
        self.xi = (f0 * sd - f1 * sr)/ rr
        self.xi_jk = (f0_jk * sd_jk - f1_jk * sr_jk)/ rr_jk
        self._xi_x = (f0 * sxd - f1 * sxr)/ rr
        self._xi_x_jk = (f0_jk * sxd_jk - f1_jk * sxr_jk)/ rr_jk
        self.r  = R
        self.mu = mu

        l = 2
        sab = 2
        self._L_mu = np.zeros(self.xi.shape)
        self._L_mu_jk = np.zeros(self.xi_jk.shape)
        i = 0
        for m in mu:
            self._L_mu[i,:] = lpmn(l, sab, m)[0][-1,-1]
            self._L_mu_jk[:,i,:] = lpmn(l, sab, m)[0][-1,-1]
            i += 1    
            
        self._factor = ((2 * l + 1)/ 2.0)*(math.factorial(l - sab)/math.factorial(l + sab))
        project_corr.__init__(self,self._L_mu*self.xi,self._L_mu_jk*self.xi_jk,self.r,self.mu,self._factor)
        self.cov_jk_norm = norm_cov(self.cov_jk)

class compute_fast_wgp2(project_corr):
    
    """
    Allows to compute the quadrupole component for galaxy-shear correlation. Eq 7 in arXiv 2307.02545. This version allows the pre-computation of the random number of pairs and then it can be executed to compute the galaxy-shear correlation varying the shapes of the galaxies in the shape catalogue.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
        dcat (treecorr.Catalog): Catalog of galaxy points.
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        rpcat (treecorr.Catalog): Catalog of random points for positions.        
        rscat (treecorr.Catalog): Catalog of random points for shapes.        
    Methods:
    -----------
        execute(new_shapes=False,shapes=[]): Allows the computation of the projected galaxy-shape correlation. It can be used giving a different shape catalogue as input.
    """    
    def __init__(self,pcat, scat, rpcat, rscat,config):
        
        # to finde duplicates
        dup = duplicate(pcat,scat,config['exact_position'],config['sky_threshold'])
        dup_random = duplicate(rpcat,rscat,config['exact_position'],config['sky_threshold'])

        self.config = config
        self.pcat = pcat
        self.rpcat = rpcat
        self.rscat = rscat
        self.dup_random = dup_random
        self.dup = dup
        self.scat = scat
        self.rr_p = []
        
        nbins = config['nbins']*config['grid_resolution']
        pi_max = config['rmax']-0.1
        npi = config['mubins']*config['grid_resolution']

        
        # arrays to store the output
        npairs_rr_jk = np.zeros((config['NPatches'], npi*nbins))
        self.npairs_rr = np.array([])

        r = np.array([])
        pi = np.array([])
        
        print('Computing normalisation factors and the number of pairs in the random catalogues...')
        # get pair-normalisation factors
        Nrep = dup.Nrep(np.ones(len(pcat.w)).astype(bool),np.ones(len(scat.w)).astype(bool))
        Nrep_rand = dup_random.Nrep(np.ones(len(rpcat.w)).astype(bool),np.ones(len(rscat.w)).astype(bool))

        NGtot = scat.sumw*self.pcat.sumw - Nrep
        RRtot = self.rpcat.sumw*self.rscat.sumw - Nrep_rand
        self.f0 = RRtot / NGtot
        
        f0_jk = np.zeros(config['NPatches'])
        
        
        # Pi bins
        Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
        r_bins  = np.logspace(np.log10(config['rmin']),np.log10(config['rmax']), nbins+1)
        xPi=(Pi[:-1]+Pi[1:])*0.5
        pibins = zip(Pi[:-1],Pi[1:])
        
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            
            r0 = max(config['rmin'],np.abs(plow))
            nbins0 = sum(r_bins >= r0)

            rr = treecorr.NNCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife') 
            rr.process(rpcat, rscat, metric='Euclidean', num_threads = config['ncores'])

            f_pairs   = lambda corrs: corrs[0].weight
            npairs_rr_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([rr], 'jackknife', func = f_pairs)

            r  = np.append(rr.rnom,r)
            pi = np.append(xPi[p]*np.ones(len(rr.rnom)),pi)
            self.npairs_rr = np.append(self.npairs_rr,rr.weight)
            
        for i in range(config['NPatches']):
            ppairs_jk = np.sum(self.pcat.w[self.pcat.patch != i])
            spairs_jk = np.sum(scat.w[scat.patch != i])
            rppairs_jk = np.sum(self.rpcat.w[self.rpcat.patch != i])        
            rspairs_jk = np.sum(self.rscat.w[self.rscat.patch != i])        
            
            NGtot_JK = ppairs_jk*spairs_jk - self.dup.Nrep(self.pcat.patch != i,scat.patch != i)
            RRtot_JK = rppairs_jk*rspairs_jk - self.dup_random.Nrep(self.rpcat.patch != i,self.rscat.patch != i) 
            RGtot_JK = spairs_jk*rppairs_jk 
    
            f0_jk[i] = RRtot_JK / NGtot_JK
        
        self.f0_JK = np.repeat(f0_jk,config['nbins']*config['mubins']).reshape(config['NPatches'],config['mubins'],config['nbins'])


        npairs_rr_jk = npairs_rr_jk[:,:len(r)]
        m = np.abs(pi/r) < 1.
        
        sum_pairs_rr_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                        npairs_rr_jk[:,m],
                                        bins=[config['mubins'],config['nbins']], 
                                        statistic = 'sum'
                                       )
        

        self.rr_jk = sum_pairs_rr_jk.statistic
        print('Everything ok :) Finished')
        
    def use_new_shapes(self,shapes):
        """
        Define new shape catalogue to compute the galaxy-shear correlation.

        Arguments
        ----------
        shapes : list
            Hosts the catalogue of the shapes (pandas array).
        Attributes:
        ----------
        scat (treecorr.Catalog): Catalog of galaxy shapes.
        """
        
        config = self.config
        
        del(self.scat)
        if config['box']:
            x, y, z, g1, g2, w = config['col_names']
            zshift = 1.e6 #to put the observer far away                   
            scat  = treecorr.Catalog(g1=shapes[g1], 
                                     g2 = shapes[g2],
                                     x = shapes[x], 
                                     y = shapes[z]+zshift,  
                                     z = shapes[y], 
                                     w = shapes[w], 
                                     patch_centers = self.pcat.patch_centers)
        
        else:   
            ra, dec, r, g1, g2, w = config['col_names']
            scat  = treecorr.Catalog(g1 = shapes[g1],
                                          g2 = shapes[g2],
                                          ra=shapes[ra], 
                                          dec=shapes[dec], 
                                          r=shapes[r], 
                                          w = shapes[w], 
                                          patch_centers = self.pcat.patch_centers, 
                                          ra_units='deg', dec_units='deg')
            
        self.scat = scat

    def execute(self):
        """
        Compute the galaxy-shear correlation.

        Attributes:
        ----------
            r (ndarray): Array of projected radial separation bins.
            mu (ndarray): Array of mu (cosine of the angle) separation bins.
            xi (ndarray): 2D galaxy-shear plus correlation function in bins of 3D radial coordinate and mu.
            xi_jk (ndarray): 2D galaxy-shear plus correlation function for each Jackknife resampling.
            xip (ndarray): Projected correlation function across mu.
            xip_jk (ndarray): Projected correlation function across mu for each Jackknife resampling.
            cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
            cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
            std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
        """
        
        config = self.config
        scat = self.scat

        nbins = config['nbins']*config['grid_resolution']
        pi_max = config['rmax']-0.1
        npi = config['mubins']*config['grid_resolution']
        
        # Pi bins
        Pi = np.linspace(-1.*pi_max, pi_max, npi+1)
        r_bins  = np.logspace(np.log10(config['rmin']),np.log10(config['rmax']), nbins+1)

        # Arrays to save results
        r = np.array([])
        pi = np.array([])

        gamma_sd = np.array([])        
        npairs_sd = np.array([])
        
        gamma_sd_jk  = np.zeros((config['NPatches'], npi*nbins))        
        npairs_sd_jk = np.zeros((config['NPatches'], npi*nbins))

        xPi=(Pi[:-1]+Pi[1:])*0.5
        pibins = zip(Pi[:-1],Pi[1:])
        # now loop over Pi bins, and compute w(r_p | Pi)
        for p,(plow,phigh) in enumerate(pibins):
            
            r0 = max(config['rmin'],np.abs(plow))
            nbins0 = sum(r_bins >= r0)
            
            sd = treecorr.NGCorrelation(nbins=nbins0, min_sep=r0, max_sep=config['rmax'], min_rpar=plow, max_rpar=phigh, bin_slop=config['slop'], brute = False, verbose=0, var_method = 'jackknife')
            
            sd.process(self.pcat, scat, metric='Euclidean', num_threads = config['ncores'])
            gamma_sd = np.append(gamma_sd,sd.xi)
            npairs_sd = np.append(npairs_sd,sd.weight)
    
    
            #Here I compute the variance
            f_pairs   = lambda corrs: corrs[0].weight
            f_gamma   = lambda corrs: corrs[0].xi
            gamma_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_gamma)        
            npairs_sd_jk[:,len(r):len(r)+nbins0], weight = treecorr.build_multi_cov_design_matrix([sd], 'jackknife', func = f_pairs)
    
            r  = np.append(sd.rnom,r)
            pi = np.append(xPi[p]*np.ones(len(sd.rnom)),pi)
                
        gamma_sd_jk  = gamma_sd_jk[:,:len(r)]        
        npairs_sd_jk = npairs_sd_jk[:,:len(r)]
        

        # Now bin over r and mu
        m = np.abs(pi/r) < 1.
        self._pi = pi
        self._r  = r
        
        all_sums = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                               [gamma_sd[m]*npairs_sd[m],
                                                self.npairs_rr[m]
                                               ], 
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
        
        sum_gamma_sd_jk = stats.binned_statistic_2d((pi/r)[m],np.log10(r[m]), 
                                                (gamma_sd_jk*npairs_sd_jk)[:,m],
                                                bins=[config['mubins'],config['nbins']], 
                                                statistic = 'sum'
                                               )
                

        # Compute the xi and JK
        mu = all_sums.x_edge[:-1] + 0.5*np.diff(all_sums.x_edge)
        R = 10**(all_sums.y_edge[:-1] + 0.5*np.diff(all_sums.y_edge))
        sd, rr = all_sums.statistic        
        sd_jk = sum_gamma_sd_jk.statistic
        
        self.xi = (self.f0 * sd)/ rr
        self.xi_jk = (self.f0_JK * sd_jk)/ self.rr_jk
        self.r  = R
        self.mu = mu

        l = 2
        sab = 2
        self._L_mu = np.zeros(self.xi.shape)
        self._L_mu_jk = np.zeros(self.xi_jk.shape)
        
        i = 0
        for m in mu:
            self._L_mu[i,:] = lpmn(l, sab, m)[0][-1,-1]
            self._L_mu_jk[:,i,:] = lpmn(l, sab, m)[0][-1,-1]
            i += 1    
            
        self._factor = ((2 * l + 1)/ 2.0)*(math.factorial(l - sab)/math.factorial(l + sab))
        
        project_corr.__init__(self,self._L_mu*self.xi,self._L_mu_jk*self.xi_jk,self.r,self.mu,self._factor)
        self.cov_jk_norm = norm_cov(self.cov_jk)
        


class get_wgx(project_corr):
    
    """
    Computes the projected cross component for galaxy-shear correlation. 
    
    Arguments:
    -----------
        xi (ndarray): 2D galaxy-shear cross correlation function in bins of 3D radial coordinate and mu.
        xi_jk (ndarray): 2D galaxy-shear cross correlation function for each Jackknife resampling.
        rcor (ndarray): Array of projected radial separation bins.        
        pcor (ndarray): Array of the upper and lower limits of the separation bins.
        L_mu_jk (optional, ndarray): Legendre polinomials.
        factor (optional,float): Factor to be included in the integration.

    Attributes:
    -----------
        r (optional, ndarray): Array of 3D radial separation bins.
        mu (optional, ndarray): Array of mu (cosine of the angle) separation bins.
        rp (optional, ndarray): Array of projected radial separation bins.
        pi (optional, ndarray): Array of l.o.s. separation bins.
        xi (ndarray): 2D galaxy-shear cross correlation function.
        xip (ndarray): Projected correlation function across the pcor direction.
        xi_jk (ndarray): 2D galaxy-shear cross correlation function for each Jackknife resampling.
        xip_jk (ndarray): Projected correlation function across pcor for each Jackknife resampling.
        cov_jk (ndarray): Covariance matrix estimated using Jackknife resampling.
        cov_jk_norm (ndarray): Normalised covariance matrix estimated using Jackknife resampling.      
        std_from_cov (ndarray): Standard deviation computed from the diagonal elements of the covariance.
    """

    
    def __init__(self,xi_x,xi_x_jk,rcor,pcor,factor=1,L_mu=1,L_mu_jk=1,q = False):
        
        project_corr.__init__(self,L_mu*xi_x,L_mu_jk*xi_x_jk,rcor,pcor,factor)
        self.xi = xi_x
        self.xi_jk = xi_x_jk
        if q:
            self.r  = rcor
            self.mu = pcor
        else:
            self.rp = rcor
            self.Pi = pcor            
        self.cov_jk_norm = norm_cov(self.cov_jk)

class compute_2p_corr():
    
    """
    Computes correlation estimators useful to measure Intrinsic Alignment.

    This class calculates the projected galaxy-galaxy (wgg), galaxy-shear (wg+, wgx),
    and the quadrupole component of the galaxy-shear (wg+2, wgx2) correlations.
    
    Arguments:
    -----------
        positions (pandas array): Catalogue of galaxy positions.
        shapes (pandas array): Catalogue of galaxy positions and shapes.
        randoms_positions (pandas array): Catalogue of randoms positions.
        randoms_shapes (pandas array): Catalogue of random positions for the shape catalogue.

    Attributes:
    -----------
        config (dict): Configuration dictionary for the computation.
        scat (treecorr.Catalog): Catalog of shapes.
        pcat (treecorr.Catalog): Catalog of positions.
        rpcat (treecorr.Catalog): Catalog of random positions.
        rscat (treecorr.Catalog): Catalog of random positions.
        wgg (object, optional): Projected galaxy-galaxy (position-position) correlation.
        wgg_cross (object, optional): Projected galaxy-galaxy (position-shape) correlation.
        wgp (object, optional): Projected galaxy-shear plus correlation.
        wgx (object, optional): Projected galaxy-shear cross correlation.
        wgp2 (object, optional): Projected quadrupole galaxy-shear plus correlation.
        wgx2 (object, optional): Projected quadrupole galaxy-shear cross correlation.
        wgp_fast (object, optional): Alternative computation of the projected galaxy-shear plus correlation.
        wgp2_fast (object, optional): Alternative computation of the projected quadrupole galaxy-shear plus correlation.

    Methods:
    -----------
        compute_wgg():
            Computes the projected galaxy-galaxy correlation.
        compute_wgg_cross():
            Computes the cross projected position-shape correlation.            
        compute_wgp():
            Computes the projected galaxy-shape plus and cross correlation.
        compute_wgp2():
            Computes the projected quadrupole galaxy-shape plus and cross correlation.
        compute_wpp():
            Computes the projected shape-shape plus and cross correlation.            
        compute_wgp_fast():
            Computes the projected galaxy-shape plus correlation. Allows the pre-computation of the number of pairs in the random catalogues and computes the correlation for different shapes catalogues.
        compute_wgp2_fast():
            Computes the projected quadrupole galaxy-shape plus correlation. Allows the pre-computation of the number of pairs in the random catalogues and computes the correlation for different shapes catalogues.
            
    """
    def __init__(self,positions,shapes,randoms_positions,randoms_shapes,config):
        
        for key in config_setup.keys():
            if not config.get(key):
                config[key] = config_setup[key]  
        
        self.config = config
        
        if config['box']:
            
            x, y, z, g1, g2, w = config['col_names']
            zshift = 1.e5 #to put the observer far away
            Nsub_box = int(round(config['NPatches'] ** (1./3)))
            # Divide the volume into 5 equal patches per axis
            patch_size =  config['box_size']/ Nsub_box  # Size of each patch
            config['NPatches'] = int(Nsub_box ** 3)
            # Compute patch indices manually
            ix = np.clip(np.floor(positions[x] / patch_size).astype(int), 0, Nsub_box - 1)
            iy = np.clip(np.floor(positions[y] / patch_size).astype(int), 0, Nsub_box - 1)
            iz = np.clip(np.floor(positions[z] / patch_size).astype(int), 0, Nsub_box - 1)

            # Compute patch ID 
            patch_id = ix + iy *Nsub_box + iz * Nsub_box**2  # Ranges from 0 to m**3 -1

            
            self.pcat  = treecorr.Catalog(x = positions[x], 
                                     y = positions[z]+zshift, 
                                     z = positions[y], 
                                     w = positions[w], 
                                     patch = patch_id)
            
            self.scat  = treecorr.Catalog(g1=shapes[g1], 
                                     g2 = shapes[g2],
                                     x = shapes[x], 
                                     y = shapes[z]+zshift,  
                                     z = shapes[y], 
                                     w = shapes[w], 
                                     patch_centers = self.pcat.patch_centers)
            
            self.rpcat = treecorr.Catalog(x=randoms_positions[x], 
                                     y=randoms_positions[z]+zshift,  
                                     z=randoms_positions[y], 
                                     patch_centers = self.pcat.patch_centers
                                     )
            

            self.rscat = treecorr.Catalog(x=randoms_shapes[x], 
                                     y=randoms_shapes[z]+zshift,  
                                     z=randoms_shapes[y], 
                                     patch_centers = self.pcat.patch_centers
                                     )

        
        else:   
            ra, dec, r, g1, g2, w = config['col_names']
                        
            self.pcat  = treecorr.Catalog(ra=positions[ra], 
                                     dec=positions[dec], 
                                     w = positions[w], 
                                     r=positions[r], 
                                     npatch = config['NPatches'], 
                                     ra_units='deg', dec_units='deg')

            self.scat  = treecorr.Catalog(g1 = shapes[g1],
                                          g2 = shapes[g2],
                                          ra=shapes[ra], 
                                          dec=shapes[dec], 
                                          r=shapes[r], 
                                          w = shapes[w], 
                                          patch_centers = self.pcat.patch_centers, 
                                          ra_units='deg', dec_units='deg')
           
            self.rpcat = treecorr.Catalog(ra=randoms_positions[ra],
                                          dec=randoms_positions[dec], 
                                          r=randoms_positions[r], 
                                          patch_centers = self.pcat.patch_centers, 
                                          ra_units='deg', dec_units='deg')

            self.rscat = treecorr.Catalog(ra=randoms_shapes[ra],
                                      dec=randoms_shapes[dec], 
                                      r=randoms_shapes[r], 
                                      patch_centers = self.pcat.patch_centers, 
                                      ra_units='deg', dec_units='deg')




    def compute_wgg(self):       
        self.wgg = compute_wgg(self.pcat,self.rpcat,self.config)

    def compute_wgg_cross(self):       
        self.wgg_cross = compute_wgg_cross(self.pcat,self.rpcat,self.scat,self.rscat,self.config)
    
    def compute_delta_sigma(self):       
        self.gs = compute_delta_sigma(self.pcat,self.scat,self.config)
        
    def compute_wgp(self):
        self.wgp = compute_wgp(self.pcat,self.scat,self.rpcat,self.rscat,self.config)
        self.wgx = get_wgx(self.wgp._xi_x,self.wgp._xi_x_jk,self.wgp.rp,self.wgp.Pi_bins)
        
    def compute_wgp2(self):
        self.wgp2 = compute_wgp2(self.pcat,self.scat,self.rpcat,self.rscat,self.config)
        
        self.wgx2 = get_wgx(self.wgp2._xi_x,self.wgp2._xi_x_jk,
                           self.wgp2.r,self.wgp2.mu,
                           self.wgp2._factor,
                           self.wgp2._L_mu_jk[0],
                           self.wgp2._L_mu_jk,
                           True)
        
    def compute_wgp_fast(self):
        self.wgp_fast = compute_fast_wgp(self.pcat,self.scat,self.rpcat,self.rscat,self.config)

    def compute_wgp2_fast(self):
        self.wgp2_fast = compute_fast_wgp2(self.pcat,self.scat,self.rpcat,self.rscat,self.config)

    def compute_wpp(self):       
        self.wpp = compute_wpp(self.scat,self.rscat,self.config)
        self.wxx = get_wgx(self.wpp._xi_x,self.wpp._xi_x_jk,
                           self.wpp.rp,self.wpp.Pi_bins)

def make_randoms_lightcone(ra, dec, z, size_random, col_names=['ra','dec','z']):

    ra_rand = np.random.uniform(min(ra), max(ra), size_random)
    sindec_rand = np.random.uniform(np.sin(min(dec*np.pi/180)), np.sin(max(dec*np.pi/180)), size_random)
    dec_rand = np.arcsin(sindec_rand)*(180/np.pi)

    y,xbins  = np.histogram(z, 50)
    x  = xbins[:-1]+0.5*np.diff(xbins)
    n = 20
    poly = np.polyfit(x,y,n)
    zr = np.random.uniform(z.min(),z.max(),1000000)
    poly_y = np.poly1d(poly)(zr)
    poly_y[poly_y<0] = 0.
    peso = poly_y/sum(poly_y)
    z_rand = np.random.choice(zr,len(ra_rand),replace=True,p=peso)

    #z_rand = np.random.choice(z,size=len(ra_rand),replace=True)

    d = {col_names[0]: ra_rand, col_names[1]: dec_rand, col_names[2]:z_rand}
    randoms = d
    #randoms = pd.DataFrame(data = d)

    return randoms


def make_randoms_box(x, y, z, size_random, col_names=['x','y','z']):
    
    val_min = x.min()
    val_max = x.max()
    xv, yv, zv = np.random.randint(val_min, val_max+1, size=(3,size_random)).astype('float')
    xv += np.random.uniform(0,1,len(xv))    
    yv += np.random.uniform(0,1,len(yv))    
    zv += np.random.uniform(0,1,len(zv))    
    
    randoms = {col_names[0]: np.clip(xv,val_min,val_max), col_names[1]: np.clip(yv,val_min,val_max), col_names[2]:np.clip(zv,val_min,val_max)}

    return randoms


