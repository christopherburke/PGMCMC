""" Module for Pretty Good MCMC parameter estimation
    Heavily based upon Philip Gregory's awesome Bayesian book
    Author: Christopher J Burke 
"""
import numpy as np
import matplotlib.pyplot as plt
from pgmcmc import pgmcmc_ioblk, pgmcmc_setup
from pgmcmc import pgmcmc_run_mcmc
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
    ioblk.calcvals[3] = np.log10(ioblk.calcvals[3])
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
    ioblk.physvals[3] = np.power(10.0, ioblk.physvals[3])
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
        #print('1 ', err)
        #print(ioblk.calcvals)
        #print(ioblk.calcval_mins)
        #print(ioblk.calcval_maxs)
        #exit()
    else:
        ioblk, err = pgmcmc_physvals(ioblk)
        # physvals must be in bounds
        anybadcheck2 = np.any(np.logical_or(ioblk.physvals < ioblk.physval_mins, \
                              ioblk.physvals > ioblk.physval_maxs))
            
        if (anybadcheck2):
            err = 1
            #print('2 ', err)
            #print(ioblk.physvals)
            #print(ioblk.physval_mins)
            #print(ioblk.physval_maxs)
            #exit()

        # Require scale1 < scale2
        if ioblk.physvals[1]+0.05 > ioblk.physvals[3]:
            err = 1
            #print('3 ', err)
    # If variables are within bounds then proceed to calculate prior
    if (err == 0):
        priorvals = np.log(1.0 / (ioblk.calcval_maxs - ioblk.calcval_mins))
        priorvals[1] = priorvals[1]*np.sqrt(2.0) # requiring scale1<scale2
        priorvals[3] = priorvals[3]*np.sqrt(2.0)  # roughly removes half of 
                            # scal1*scale2 area out so divide each by sqrt(2)
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
        shape1 = ioblk.physvals[0]
        scale1 = ioblk.physvals[1]
        shape2 = ioblk.physvals[2]
        scale2 = ioblk.physvals[3]
        p1 = ioblk.physvals[4]
        ypdf = p1*weibull_min.pdf(ioblk.xData, shape1, 0.0, scale1) + (1.0-p1)*weibull_min.pdf(ioblk.xData, shape2, 0.0, scale2)

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
    NSamp = 2000
    betPar1 = 1.5
    etaPar1 = 50.0
    betPar2 = 3.0
    etaPar2 = 200.0
    P1 = 1.0/3.0
    
    obs = np.zeros((NSamp,), dtype=float)
    # First do binomial distribution pulls to assign 1 or 0 as to whether
    #  part belongs to 1st or 2nd weibull distribution
    binpulls = np.random.binomial(1, P1, size=(NSamp,))
    idxP1 = np.where(binpulls == 1)[0]
    idxP2 = np.where(binpulls == 0)[0]
    nP1 = len(idxP1)
    nP2 = len(idxP2)
    print('Nsamp: {0:d} P1: {1:f} N1: {2:d} N2: {3:d}'.format(NSamp, P1, nP1, nP2))
    obs[idxP1] = np.random.weibull(betPar1, size=(nP1,)) * etaPar1
    obs[idxP2] = np.random.weibull(betPar2, size=(nP2,)) * etaPar2

    minObs = np.min(obs)
    maxObs = np.max(obs)
    mnObs = np.mean(obs)
    print('Obs min: {0:f} mn: {1:f} mx: {2:f}'.format(minObs, mnObs, maxObs))
    
    #Instantiate pgmcmc_ioblk class and fill in initial values
    ioblk = pgmcmc_ioblk()
    # Starting values and limits
    ioblk.physval_names = np.array(['shape1','scale1','shape2','scale2','P1'])
    ioblk.calcval_names = np.array(['c_shape1', 'log_scale1','c_shape2','log_scale2','c_P1'])
    # origests are 
    ioblk.origests = np.array([1.0, mnObs/1.5, 1.0, mnObs*1.5, 0.5], dtype=float)

    ioblk.fixed = np.zeros_like(ioblk.origests, dtype=int)
    ioblk.physval_mins = np.array([0.1, 0.01, 0.1, 0.01, 0.05], dtype=float)
    ioblk.physval_maxs = np.array([10.0, 1000.0, 10.0, 1000.0, 0.95], dtype=float)
    ioblk.calcval_mins = np.array([0.1, -2.0, 0.1, -2.0, 0.05], dtype=float)
    ioblk.calcval_maxs = np.array([10.0, 3.0, 10.0, 3.0, 0.95], dtype=float)
    ioblk.scls = np.array([0.1, 0.2, 0.1, 0.2, 0.05], dtype=float)

    ioblk.xData = obs

    
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
    #print(ioblk.physvals)
    #print(ioblk.physval_mins)
    #print(ioblk.physval_maxs)
    #print(ioblk.calcvals)
    #print(ioblk.calcval_mins)
    #print(ioblk.calcval_maxs)
#    
#    # Setup and passed all checks ready to start 

    ioblk = pgmcmc_run_mcmc(ioblk)
