""" Module for Pretty Good MCMC parameter estimation
    Heavily based upon Philip Gregory's awesome Bayesian book
    Author: Christopher J Burke 
"""
import numpy as np
import matplotlib.pyplot as plt
from pgmcmc import pgmcmc_ioblk, pgmcmc_setup
from pgmcmc import pgmcmc_run_mcmc, pgmcmc_run_minimizer
from scipy.stats import weibull_min

def pgmcmc_calcvals(ioblk):
    """Convert physical parameters to calculation variables
       This is where you scale physical variables or more generic
         variable transformation
       In this example we are not doing any variable transformations
       INPUT:
         ioblk - [class] pgmcmc_ioblk class
       OUTPUT: 
         ioblk - [class]
         err - [0 ok ; 1 not ok]
    """
    err = 0 # Error flag
    ioblk.calcvals = np.copy(ioblk.physvals)
    ioblk.calcvals[1] = np.log10(ioblk.calcvals[1])
    if ~np.isfinite(ioblk.calcvals).all():
        print("Calculation Values Bad")
        print(ioblk.calcvals)
        print(ioblk.physvals)                 
        err = 1
    return ioblk, err

def pgmcmc_physvals(ioblk):
    """Convert calculation variables to physical variables
    This is where you undo the variable transformation performed in
    pgmcmc_calcvals.
    In this example there is not any variables transformations
       INPUT:
         ioblk - [class] pgmcmc_ioblk class
       OUTPUT: 
         ioblk - [class]
         err - [0 ok ; 1 not ok]
    """
    err = 0 # Error flag
    ioblk.physvals = np.copy(ioblk.calcvals)
    ioblk.physvals[1] = np.power(10.0, ioblk.physvals[1])
    if ~np.isfinite(ioblk.physvals).all():                 
        print("Physical Values Bad")
        print(ioblk.calcvals)
        print(ioblk.physvals)                
        err = 1
    return ioblk, err

def pgmcmc_prior(ioblk):
    """Calculate prior and also check variable bounds
       This function must catch all variable values that would
       cause the model or likelihood to fail, nan, imaginary, or infs
    """
    err = 0 # Error flag
    prior = 0.0
    ioblk.calcvals[ioblk.mcmc.paridx] = ioblk.mcmc.pars
    # First the calcvals must be in bounds
    anybadcheck = np.any(np.logical_or(ioblk.calcvals < ioblk.calcval_mins, \
                         ioblk.calcvals > ioblk.calcval_maxs))
    if (anybadcheck):
        err = 1
    else:
        ioblk, err = pgmcmc_physvals(ioblk)
        # physvals must be in bounds
        anybadcheck2 = np.any(np.logical_or(ioblk.physvals < ioblk.physval_mins, \
                              ioblk.physvals > ioblk.physval_maxs))
        if (anybadcheck2):
            err = 1

    # If variables are within bounds then proceed to calculate prior
    if (err == 0):
        priorvals = np.log(1.0 / (ioblk.calcval_maxs - ioblk.calcval_mins))
        prior = np.sum(priorvals)
        # Warn on nonfinite and nonreal prior 
        if ~np.isfinite(prior):
            print("Non Finite Prior")
            print(ioblk.physvals)
        if ~np.isreal(prior):
            print("Non Real Prior")
            print(ioblk.physvals)
    ioblk.mcmc.prior = prior
    return ioblk, err    

def pgmcmc_likehood(ioblk):
    """Calculate likelihood
       This returns the log likelihood given a model
       We defer error and parameter bound checking on values in the pgmcmc_prior
       routine.
    """
    err = 0 # Error flag
    loglike = 0.0
    chi2 = 0.0
    

    # Model is valid calculate likelihood
    if (err == 0):
        shape = ioblk.physvals[0]
        scale = ioblk.physvals[1]
        ypdf = np.zeros_like(ioblk.xData)
        # objects that failed get weibull pdf
        ypdf[ioblk.xFidx] = weibull_min.pdf(ioblk.xData[ioblk.xFidx], shape, 0.0, scale)
        # objects that did not fail are censored get reliability 1-cdf
        ypdf[ioblk.xNFidx] = 1.0 - weibull_min.cdf(ioblk.xData[ioblk.xNFidx], shape, 0.0, scale)

        loglike = np.sum(np.log(ypdf))
    ioblk.mcmc.like = loglike
    ioblk.mcmc.chi2 = chi2
    
    if (ioblk.parm.debugLevel > 2 and \
        ioblk.mcmc.curpar == 0 and \
        ioblk.pt.curtempidx == ioblk.pt.ntemp - 1 and \
        np.mod(ioblk.mcmc.pos,ioblk.parm.likehoodmoddisplay) == 0):
        
        pgmcmc_showmodel(ioblk)
        
    return ioblk, err    

def pgmcmc_showmodel(ioblk):
    plt.figure(ioblk.fighandle.number)
    plt.cla()
    plt.ecdf(ioblk.xData)

    plt.pause(1.0e-4)
    if ioblk.parm.debugLevel > 3:
        input("Press [Enter]")
    return

    
# Run the mcmc
if __name__ == "__main__":
    NSamp = 100
    betPar = 1.5
    etaPar = 8.0
    censorPrc = 0.8 # 0.1 means treat highest 10% of observations as
                    # not failed yet
    

    # Generate weibull data
    #  betPar is the shape parameter k
    # etaPar is the scale factor  - apply scale factor after generating samples
    obs = np.random.weibull(betPar, size=(NSamp,)) * etaPar
    minObs = np.min(obs)
    maxObs = np.max(obs)
    mnObs = np.mean(obs)
    
    # Do censoring of censorPrc
    obsSort = np.sort(obs)
    idxUL = int(NSamp * (1.0 - censorPrc))
    valUL = obsSort[idxUL]
    idxFail = np.where(obs < valUL)[0]
    idxNotFail = np.where(obs >= valUL)[0]
    nFail = len(idxFail)
    nNotFail = len(idxNotFail)
    print('NSamp: {0:d} valUL: {1:f} nFail: {2:d} nNotFail: {3:d}'.format(NSamp, valUL, nFail, nNotFail))
    # change observed failure rate to the upper limit
    obs[idxNotFail] = valUL

    #Instantiate pgmcmc_ioblk class and fill in initial values
    ioblk = pgmcmc_ioblk()
    # Starting values and limits
    ioblk.physval_names = np.array(['shape','scale'])
    ioblk.calcval_names = np.array(['c_shape', 'log_scale'])
    ioblk.origests = np.array([1.0, mnObs], dtype=float)

    ioblk.fixed = np.zeros_like(ioblk.origests, dtype=int)
    ioblk.physval_mins = np.array([0.1, 0.01], dtype=float)
    ioblk.physval_maxs = np.array([10.0, 1000.0], dtype=float)
    ioblk.calcval_mins = np.array([0.1, -2.0], dtype=float)
    ioblk.calcval_maxs = np.array([10.0, 3.0], dtype=float)
    ioblk.scls = np.array([0.1, 0.2], dtype=float)

    ioblk.xData = obs
    ioblk.xFidx = idxFail
    ioblk.xNFidx = idxNotFail
    
#    # Assign the users problem specific functions
    ioblk.func_physvals = pgmcmc_physvals
    ioblk.func_calcvals = pgmcmc_calcvals
    ioblk.func_prior = pgmcmc_prior
    ioblk.func_likehood = pgmcmc_likehood
    ioblk.func_showmodel = pgmcmc_showmodel
    ioblk.parm.debugLevel = 3
    ioblk.parm.likehoodmoddisplay = 1000
    ioblk.likecount = 0
#    
#    # Turn on parallel tempering and set temperatures
    ioblk.parm.dopartemp = False
    ioblk.pt.temps = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0])
    ioblk.parm.saveAltTemp = False
    ioblk.parm.savedTempIdx = 0
#    
#    # Setup other things
    ioblk.parm.debugLevel=2
    ioblk.parm.maxstps = 100000
    ioblk = pgmcmc_setup(ioblk)
#    
#    # Setup and passed all checks ready to start 

    ioblk = pgmcmc_run_mcmc(ioblk)
