# PGMCMC
Pretty Good MCMC with Parallel Tempering
Modeled after the descriptions of MCMC parameter estimation techniques from Phil Gregory's excellent book
Bayesian Logical Data Analysis for the Physical Sciences: A Comparative Approach with Mathematica Support

No Real Documentation as of yet, but 
ipython pgmcmc_user.py 
will run the test case of fitting a straight line to data where the uncertainties have been underestimated.
It includes the data uncertainty scale as a free parameter.
This routine will periodically save the current state of the run in an HDF5 file

You can see the results of the run through
ipython pgmcmc_report.py 
This will show the parameter distribution for each parameter, the parameter through the chain, its ACF
Also displays 2D parameter correlation distributions.

The nuts and bolts main code that is independent of the problem being solved is in pgmcmc.py

If you want to use these routines for your own problem, then you need to basically copy the functionality
of all routines in pgmcmc_user.py
In the example problem the parallel tempering does actually need to be used, but it is turned on 
just to illustrate how to use it.  Also the conversion from physical variables to the variables used in the
calculation is redundant for this problem, but the functions do need to exist in order to handle more
complicated problems where you do want to reparameterize for the problem.
