""" Module for Pretty Good MCMC parameter estimation
    Heavily based upon Philip Gregory's awesome Bayesian book
    Author: Christopher J Burke 
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
from pgmcmc import pgmcmc_ioblk, pgmcmc_parameters, pgmcmc_mcmc, pgmcmc_setup
from pgmcmc import pgmcmc_run_mcmc, pgmcmc_iterate_proposals, pgmcmc_save_state
from pgmcmc import pgmcmc_burnit
from pgmcmc import press_key_to_close_figure, on_key_event
       
            
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
        priorvals = np.log(1.0 / (ioblk.physval_maxs - ioblk.physval_mins))
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
    
    # Users model
    ioblk, err = pgmcmc_model(ioblk)
    
    # Model is valid calculate likelihood
    if (err == 0):
        errscl = ioblk.physvals[2]
        resids = (ioblk.yData - ioblk.yModel) / ioblk.errData
        chi2 = np.sum(resids * resids)
        if (chi2 < ioblk.chi2min):
            print("New Best Chi2: {0:f} Old: {1:f} Expected: {2:f}".format( \
                 chi2, ioblk.chi2min, ioblk.expchi2))
            ioblk.chi2min = chi2
            ioblk.bestphysvals = ioblk.physvals
            ioblk.bestcalcvals = ioblk.calcvals
        ndat = ioblk.xData.size
        loglike = ioblk.likenorm - (ndat * np.log(errscl)) - \
                      (chi2/2.0/errscl/errscl)
    ioblk.mcmc.like = loglike
    ioblk.mcmc.chi2 = chi2
    
    if (ioblk.parm.debugLevel > 2 and \
        ioblk.mcmc.curpar == 0 and \
        ioblk.pt.curtempidx == ioblk.pt.ntemp - 1 and \
        np.mod(ioblk.mcmc.pos,ioblk.parm.likehoodmoddisplay) == 0):
        
        pgmcmc_showmodel(ioblk)
        
    return ioblk, err    

def pgmcmc_model(ioblk):
    """ Model being fit to data
        In this case it is a line
    """
    err = 0
    constant = ioblk.physvals[0]
    slope = ioblk.physvals[1]
    ioblk.yModel = constant + (slope * ioblk.xData)
    if ~np.isfinite(ioblk.yModel).all():                 
        print("Model Values nonfinite")
        print(ioblk.calcvals)
        print(ioblk.physvals)               
        err = 1
    if ~np.isreal(ioblk.yModel).all():                 
        print("Model Values Imaginary")
        print(ioblk.calcvals)
        print(ioblk.physvals)                 
        err = 1
    return ioblk, err
        
def pgmcmc_showmodel(ioblk):
    plt.figure(ioblk.fighandle.number)
    plt.cla()
    plt.errorbar(ioblk.xData, ioblk.yData, yerr=ioblk.errData, fmt='ok', \
                 markersize=2.0)
    plt.plot(ioblk.xData, ioblk.yModel, '-r')
    plt.pause(1.0e-4)
    if ioblk.parm.debugLevel > 3:
        raw_input("Press [Enter]")
    
# Run the test case of a linear fit with an uncertain relative errobar
if __name__ == "__main__":
    # Set up the test case of a linear relationship
    # Line has slope and constant
    realConst = -1.0
    realSlope = 2.14
    realErrorStdDev = 2.0
    nPts = 100

    # Independent variables are uniformly randomly distributed
    #  between -1<x<+1
    xValues = np.random.random_sample((nPts,)) * 2.0 - 1.0
    # Dependent variables follow linear line + gaussian error
    yValues = realConst + (realSlope * xValues) + \
                 (realErrorStdDev * np.random.randn(nPts))
    yActualValues = realConst + (realSlope * xValues)
    #  Assume the uncertainties on data points is 1.0
    errValues = np.full_like(xValues, 1.0)

    # Make a figure of the data
    #fig, ax, fsd = press_key_to_close_figure()
    #ax.errorbar(xValues, yValues, yerr=errValues, fmt='ok')
    #ax.plot(xValues, yActualValues, '--k')
    #plt.show()

    #Instantiate pgmcmc_ioblk class and fill in initial values
    ioblk = pgmcmc_ioblk()
    # In the linear line fit problem with unknown relative
    #  error bar scaling there are 3 parameters 
    #  line constant and slope plus error scaling factor
    # Give variable text names used in plotting
    ioblk.physval_names = ['Const', 'Slope', 'ErrScale']
    ioblk.calcval_names = ['Const', 'Slope', 'ErrScale']
    # Give seed starting values for the mcmc
    ioblk.origests = np.array([np.mean(yValues), 1.0, 1.0])
    # Give integer array for variables you want fixed during fit
    # 0 - not fixed (solved for) ; 1 - fixed (not solved for)
    ioblk.fixed = np.array([0, 0, 0])
    # Give upper and lower limits of variables
    #  Should comfortably constrain values
    #  and in this case will be used as hard constraints in the prior
    #  allowed range 
    ioblk.physval_mins = np.array([-10.0, -10.0, 1.0e-3])
    ioblk.physval_maxs = np.array([10.0, 10.0, 1.0e3])
    ioblk.calcval_mins = ioblk.physval_mins
    ioblk.calcval_maxs = ioblk.physval_maxs
    # Now specify the initial step size for parameter jumps.
    # these only have to be approximately correct as they will be 
    # iteratively solved for later
    ioblk.scls = np.array([1.0, 0.5, 0.3])
    # Add in data that will be fit
    ioblk.yData = yValues
    ioblk.errData = errValues
    ioblk.xData = xValues
    # Assign the users problem specific functions
    ioblk.func_physvals = pgmcmc_physvals
    ioblk.func_calcvals = pgmcmc_calcvals
    ioblk.func_prior = pgmcmc_prior
    ioblk.func_likehood = pgmcmc_likehood
    ioblk.func_showmodel = pgmcmc_showmodel
    ioblk.parm.debugLevel = 3
    ioblk.parm.likehoodmoddisplay = 1000
    
    # Turn on parallel tempering and set temperatures
    ioblk.parm.dopartemp = True
    ioblk.pt.temps = np.array([0.01, 0.05, 0.1, 0.3, 0.6, 0.8, 1.0])
    ioblk.parm.saveAltTemp = True
    ioblk.parm.savedTempIdx = 0
    
    # Setup other things
    ioblk = pgmcmc_setup(ioblk)
    
    # Setup and passed all checks ready to start MCMC
    # Actually run mcmc
    pgmcmc_run_mcmc(ioblk)
