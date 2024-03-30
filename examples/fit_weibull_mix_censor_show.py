"""  Module for Pretty Good MCMC parameter estimation
    Heavily based upon Philip Gregory's awesome Bayesian book
    Author: Christopher J Burke 
    These are routines to make diagnostic plots 
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
from pgmcmc import pgmcmc_ioblk, pgmcmc_parameters, pgmcmc_mcmc
import cjb_utils as cjb
from scipy.stats import weibull_min
from scipy.special import gamma

            
def pgmcmc_load_state(prefix):
    f = h5py.File(prefix+'.hd5','r')
    pvals = np.array(f['Pvals'])
    cvals = np.array(f['Cvals'])
    bvals = np.array(f['Bvals'])
    
    ioblk = pickle.load(open(prefix+'.pkl', 'rb'))
    return ioblk, pvals, cvals, bvals

if __name__ == "__main__":
#    inputPre = 'run_0p50_singleweib'
#    outPre = 'wb_censor_0p50_singleweib'
    inputPre = 'run_0p20'
    outPre = 'wb_censor_0p20'

    betPar1 = 1.5
    etaPar1 = 50.0
    betPar2 = 3.0
    etaPar2 = 200.0
    P1 = 1.0/3.0
    censorPrc = 0.2
    
    NBin = 40
    
    ioblk, pvals, cvals, bvals  = pgmcmc_load_state(inputPre)
    names = np.array(ioblk.physval_names)
    cnames = np.array(ioblk.calcval_names)

    nfreepar = ioblk.mcmc.paridx.size
    Clipout = 200
    currentPos = ioblk.mcmc.pos
    
    shape1s = pvals[Clipout:currentPos-1,0]
    scale1s = pvals[Clipout:currentPos-1,1]
    shape2s = pvals[Clipout:currentPos-1,2]
    scale2s = pvals[Clipout:currentPos-1,3]
    p1s = pvals[Clipout:currentPos-1,4]

    lnlike = bvals[Clipout:currentPos-1,0]
    lnprior = bvals[Clipout:currentPos-1,1]
    ia = np.argmax(lnlike)
    lik = lnlike[ia]
    pri = lnprior[ia]
    nfreepar = ioblk.mcmc.paridx.size
    print('nfreepar: {0:d}'.format(nfreepar))
    aic = 2.0*nfreepar - 2.0*lik
    print('max lnLike: {0:f} lnPrior: {1:f} aik: {2:f}'.format(lik, pri, aic))

#    shape2s = shape1s
#    scale2s = scale1s
#    p1s = np.ones_like(shape1s)

    nStp = len(shape1s)

    tmp = gamma(1.0 + 1.0/shape1s)
    expMn1 = tmp*scale1s
    expVar1 = scale1s*scale1s*(gamma(1.0+2.0/scale1s) - tmp*tmp)
    tmp = gamma(1.0 + 1.0/shape2s)
    expMn2 = tmp*scale2s
    expVar2 = scale2s*scale2s*(gamma(1.0+2.0/scale2s) - tmp*tmp)

    plt.plot(expMn1, expVar1, '.')
    plt.show()
    plt.plot(expMn2, expVar2, '.')
    plt.show()
    
    obs = ioblk.xData
    print(obs[0:10])
    idxF = ioblk.xFidx
    idxNF = ioblk.xNFidx
    NSamp = len(obs)
    NFail = len(idxF)
    NNFail = len(idxNF)
    print('NSamp: {0:d} Nfail: {1:d} NNotFail {2:d}'.format(NSamp, NFail, NNFail))
    
    # From the given parameters of weibull find the 0.005 and 0.995 percentiles
    ex1s = weibull_min.ppf([0.005,0.995], betPar1, 0.0, etaPar1)
    ex2s = weibull_min.ppf([0.005,0.995], betPar2, 0.0, etaPar2)
    minObs = np.min([ex1s[0], ex2s[0]])
    maxObs = np.max([ex1s[1], ex2s[1]])*1.2
    xvals = np.linspace(minObs, maxObs, NBin*20)
    ypdf = P1*weibull_min.pdf(xvals, betPar1, 0.0, etaPar1)
    ypdf = ypdf + (1.0-P1)*weibull_min.pdf(xvals, betPar2, 0.0, etaPar2)
    ycdf = P1*weibull_min.cdf(xvals, betPar1, 0.0, etaPar1)
    ycdf = ycdf + (1.0-P1)*weibull_min.cdf(xvals, betPar2, 0.0, etaPar2)
    
    # 
    useObs = obs[idxF]
    # Show histogram of Failed observations

    fig, ax, fsd = cjb.setup_figure()
    fsd['tickfontsize'] /= 1.2
    tedges = np.linspace(minObs,maxObs, NBin-1)
    midt = tedges[:-1] + np.diff(tedges)/2.0
    n, xedgesused, patches = plt.hist(obs, tedges, histtype='bar', \
                            edgecolor=fsd['myblack'], color=fsd['myblue'], \
                            density=True)
    plt.plot(xvals, ypdf, '-', color=fsd['myblack'], linewidth=3.0)
    
    plt.xlabel('Failure Times', fontsize=fsd['labelfontsize'], fontweight='heavy')
    plt.ylabel('Distribution', fontsize=fsd['labelfontsize'], 
           fontweight='heavy')
    plt.title('Fail/Not Fail Sample Sizes {0:d}/{1:d} Censored {2:3.0f}%'.format(NFail, NNFail,censorPrc*100.0),\
              fontsize=fsd['labelfontsize']/1.1, 
           fontweight='heavy')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(fsd['plotboxlinewidth'])
        ax.spines[axis].set_color(fsd['mynearblack'])
        ax.tick_params('both', labelsize=fsd['tickfontsize'], width=fsd['plotboxlinewidth'], 
               color=fsd['mynearblack'], length=fsd['plotboxlinewidth']*3)
    plt.subplots_adjust(left=0.17, bottom=0.15)
    plt.savefig(outPre+'_obsdist.png', dpi=400)
    plt.show()
    
    NLines = 200
    lnidx = np.random.choice(nStp, size=(NLines,))
    # Show cumulative distributions
    fig, ax, fsd = cjb.setup_figure()
    fsd['tickfontsize'] /= 1.2
    plt.ecdf(obs, linestyle='-', color=fsd['myblue'], linewidth=8.0, label='Empirical CDF', zorder=5)
    plt.plot(xvals, ycdf, '--', color=fsd['myblack'], linewidth=5.0, label='Actual CDF', zorder=10)

    # Do the cdf samples from chain
    for i in range(NLines):
        k = lnidx[i]
        ycdf = p1s[k]*weibull_min.cdf(xvals, shape1s[k], 0.0, scale1s[k])
        ycdf = ycdf + (1.0-p1s[k])*weibull_min.cdf(xvals, shape2s[k], 0.0, scale2s[k])
        useLabel = 'MCMC Sample'
        if i > 0:
            useLabel = None
        plt.plot(xvals, ycdf, '-', color=fsd['myred'], linewidth=2.0, label=useLabel, alpha=0.3, zorder=0)
        


    plt.legend(fontsize=fsd['tickfontsize'])    
    plt.xlabel('Failure Times', fontsize=fsd['labelfontsize'], fontweight='heavy')
    plt.ylabel('CDF', fontsize=fsd['labelfontsize'], 
           fontweight='heavy')
    plt.title('Fail/Not Fail Sample Sizes {0:d}/{1:d} Censored {2:3.0f}%'.format(NFail, NNFail,censorPrc*100.0),\
              fontsize=fsd['labelfontsize']/1.1, 
           fontweight='heavy')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(fsd['plotboxlinewidth'])
        ax.spines[axis].set_color(fsd['mynearblack'])
        ax.tick_params('both', labelsize=fsd['tickfontsize'], width=fsd['plotboxlinewidth'], 
               color=fsd['mynearblack'], length=fsd['plotboxlinewidth']*3)
    plt.subplots_adjust(left=0.17, bottom=0.15)
    plt.savefig(outPre+'_cdf.png', dpi=400)
    plt.show()
    

    # Show probability distributions
    fig, ax, fsd = cjb.setup_figure()
    fsd['tickfontsize'] /= 1.2
    tedges = np.linspace(minObs,maxObs, NBin-1)
    midt = tedges[:-1] + np.diff(tedges)/2.0
    n, xedgesused, patches = plt.hist(obs, tedges, histtype='bar', \
                            edgecolor=fsd['myblack'], color=fsd['myblue'], \
                            density=True)
    plt.plot(xvals, ypdf, '-', color=fsd['myblack'], linewidth=3.0)

    # Do pdf samples from chains
    for i in range(NLines):
        k = lnidx[i]
        ypdf = p1s[k]*weibull_min.pdf(xvals, shape1s[k], 0.0, scale1s[k])
        ypdf = ypdf + (1.0-p1s[k])*weibull_min.pdf(xvals, shape2s[k], 0.0, scale2s[k])
        useLabel = 'MCMC Sample'
        if i > 0:
            useLabel = None
        plt.plot(xvals, ypdf, '-', color=fsd['myred'], linewidth=2.0, label=useLabel, alpha=0.3, zorder=0)


    plt.legend(fontsize=fsd['tickfontsize'])    
    plt.xlabel('Failure Times', fontsize=fsd['labelfontsize'], fontweight='heavy')
    plt.ylabel('Distribution', fontsize=fsd['labelfontsize'], 
           fontweight='heavy')
    plt.title('Fail/Not Fail Sample Sizes {0:d}/{1:d} Censored {2:3.0f}%'.format(NFail, NNFail,censorPrc*100.0),\
              fontsize=fsd['labelfontsize']/1.1, 
           fontweight='heavy')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(fsd['plotboxlinewidth'])
        ax.spines[axis].set_color(fsd['mynearblack'])
        ax.tick_params('both', labelsize=fsd['tickfontsize'], width=fsd['plotboxlinewidth'], 
               color=fsd['mynearblack'], length=fsd['plotboxlinewidth']*3)
    plt.subplots_adjust(left=0.17, bottom=0.15)
    plt.savefig(outPre+'_pdf.png', dpi=400)
    plt.show()

