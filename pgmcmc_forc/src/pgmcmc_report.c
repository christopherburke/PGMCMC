//
//  pgmcmc_report.c
//
//  Created by Christopher Burke on 8/30/22.
//

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rstat.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_histogram.h>
#include <math.h>
#include <plplot/plplot.h>
#include <pgmcmc.h>
#include <hdf5.h>

void  setclr(unsigned char *clr) {
  plscol0(15, clr[0], clr[1], clr[2]); // colormap entry 15 to drwclor
  plcol0(15); // switch to drwclr
}



void single_hist_box(double x0, double x1, double y0, double y1, unsigned char *fillclr, unsigned char *edgeclr) {
  double x[4], y[4];
  
  x[0] = x0;
  y[0] = y0;
  x[1] = x0;
  y[1] = y1;
  x[2] = x1;
  y[2] = y1;
  x[3] = x1;
  y[3] = y0;
  // Fill rectangle
  plscol0(15, fillclr[0], fillclr[1], fillclr[2]); // colormap entry 15 to drwclor
  plcol0(15); // switch to drwclr for drawing axis
  plfill(4, x, y);
  // Draw edges
  plscol0(15, edgeclr[0], edgeclr[1], edgeclr[2]); // colormap entry 15 to drwclor
  plcol0(15); // switch to drwclr for drawing axis
  plline(4, x, y);
}

void parameter_diagnostics(char * curname, gsl_vector *curDatas, gsl_vector *curLikes, gsl_vector *curPriors, gsl_vector *curChi2s, size_t nUse, int stride) {
  
  double percents[7] = {0.00135, 0.02275, 0.15865, 0.5, 0.84135, 0.97725, 0.99865};
  double limits[6] = {0.0027, 0.0455, 0.3173, 0.6827, 0.9545, 0.9973};
  double pres[7], lres[6], datamin, datamax, ubx, x0, x1, y0, y1;
  gsl_vector *sortdata, *unviewdata;
  gsl_histogram *hist_data;
  int tmpi, i;
  int NBINS = 30, NLAGS=30;
  size_t ii, j, newn;
  char label_buff[100], label_buff2[50];
  gsl_vector *stepInc;
  gsl_vector *xtmp, *ytmp;
  
  // Parameters for plotting and formatting
  char *plot_dev = "pngcairo";
  //Plotting formatting
  //const char *symbout = "\xE2\x97\x8B"; // UTF-8 circle
  //const char *symbout = "\xE2\xAD\x98"; // heavy circle
  const char *symbfil = "\xE2\x97\x8F"; // black circle
  // PNG output
  double dpi = 400.0; double xWidthIn = 6.0; double yWidthIn = 8.0; // dpi, x axis width in inches
  char *xLabel, *yLabel, *pltTitle;
  PLINT label_family = PL_FCI_SANS; PLINT label_style=PL_FCI_UPRIGHT; PLINT label_weight=PL_FCI_BOLD;
  PLINT title_family = PL_FCI_SANS; PLINT title_style=PL_FCI_UPRIGHT; PLINT title_weight=PL_FCI_BOLD;
  double labelScl = 1.5;
  double titleScl = 0.9;
  double tickScl = 1.0;
  double resultScl = 0.7;
  double mrkScl = 0.2;
  double boxWidth = 3.0;
  double edgebuff_fracx = 0.05; // fraction of data width to add to edges of axis as buffer
  double edgebuff_fracy = 0.05;
  unsigned char bkgclr[] = {0xff, 0xff, 0xff}; // White background
  unsigned char drwclr[] = {0, 0, 0}; // black color
  unsigned char fillclr[] = {61, 156, 235}; // light blue
  double vxmin, vxmax, vymin, vymax, delvx, delvy;
  int totnr, totnc, subnr, subnc, strtr, strtc;

  // for non GSL calls Need vector that is not a view
  unviewdata = gsl_vector_alloc(nUse);
  tmpi = gsl_vector_memcpy(unviewdata, curDatas);

  // Need data sorted to calculate percentiles
  sortdata = gsl_vector_alloc(nUse);
  tmpi = gsl_vector_memcpy(sortdata, curDatas);
  gsl_sort_vector(sortdata);
  for (i=0; i<7; i++){
    pres[i] = gsl_stats_quantile_from_sorted_data(gsl_vector_ptr(sortdata,0), 1, nUse, percents[i]);
    if (i<6) {
      lres[i] = gsl_stats_quantile_from_sorted_data(gsl_vector_ptr(sortdata,0), 1, nUse, limits[i]);
    }
  }
  
  // Now lets try to make a figure of the data using plplot
  plscolbg(bkgclr[0], bkgclr[1], bkgclr[2]); // set background color
  plsdev(plot_dev); // Set output device
  // Make outputfilename
  strcpy(label_buff, curname);
  plsfnam(strcat(label_buff, "_report.png")); // plot output file name
  // ***PNG SET DPI***
  plspage(dpi, dpi, lround(dpi*xWidthIn), lround(dpi*yWidthIn), 0, 0);
  plinit(); // initialize plot
  pladv(1); // this subpage advance seems needed if using plgvpd to get viewport coordinates
  
  setclr(drwclr);
  plsfont(label_family, label_style, label_weight); // font format
  plwidth(boxWidth); // set box line width

  // Set the standard viewport that makes biggest viewport leaving room for labels
  plvsta();
  plgvpd(&vxmin, &vxmax, &vymin, &vymax); // return this standard viewport rectangle given in normalized coordinates
  //printf("%lf %lf %lf %lf\n", vxmin, vxmax, vymin, vymax);
  // In our case we want a 7 row x 1 column total blocks A=3rx1c; b=1rx1c; c=1rx1c; *=space for labels
  // A
  // A
  // A
  // *
  // B
  // *
  // C
  totnr = 7;
  totnc = 1;
  delvx = (vxmax-vxmin) / ((double) totnc);
  delvy = (vymax-vymin) / ((double) totnr);
  // First plot, A, is histogram of samples
  subnr = 3;
  subnc = 1;
  strtr = 4; // start row 0 is bottom
  strtc = 0; // start column 0 is left
  // Set the viewport for A
  plvpor(vxmin+strtc*delvx, vxmin+(strtc+subnc)*delvx, vymin+strtr*delvy, vymin+(strtr+subnr)*delvy);
  // do the data histogram
  // Get the min and max of the data vector
  datamin = gsl_vector_get(sortdata, 0);
  datamax = gsl_vector_get(sortdata, nUse-1);
  ubx = fabs((datamax - datamin) * edgebuff_fracx); // buffer at edges of data window
  // use gsl to calculate histogram data
  hist_data = gsl_histogram_alloc(NBINS);
  gsl_histogram_set_ranges_uniform(hist_data, datamin-ubx, datamax+ubx);
  for (ii=0; ii<nUse; ii++) {
    gsl_histogram_increment(hist_data, gsl_vector_get(sortdata,ii));
  }
  plwind(datamin-ubx, datamax+ubx, 0.0, gsl_histogram_max_val(hist_data)*(1.0+edgebuff_fracy));
  plschr(0.0, tickScl); // scale ticklabels
  plbox("bcnts", 0.0, 0.0, "bcntsv", 0.0, 0.0);
  plschr(0.0, 1.0/tickScl); // undo ticklabel scale
  // Draw histogram one box at a time
  for (i=0; i<NBINS; i++) {
    gsl_histogram_get_range(hist_data, i, &x0, &x1);
    y0=0.0;
    y1=gsl_histogram_get(hist_data, i);
    single_hist_box(x0, x1, y0, y1, fillclr, drwclr);
  }
  // Labels
  plschr(0.0, labelScl); // scale labels
  setclr(drwclr);
  plsfont(label_family, label_style, label_weight); // font format
  plmtex("b", 2.0, 0.5, 0.5, curname); // x-axis
  plmtex("l", 2.5, 0.5, 0.5, "Counts");
  plschr(0.0, 1.0/labelScl); // undo scale labels
  // Title write out result
  plschr(0.0, titleScl); // scale title
  plsfont(title_family, title_style, title_weight); // font format
  label_buff[0] = '\0';
  strcpy(label_buff, "Median ");
  strcat(label_buff, curname);
  strcat(label_buff, " = ");
  sprintf(label_buff2, "%g +%g / -%g", pres[3], pres[4]-pres[3], pres[3]-pres[2]);
  strcat(label_buff, label_buff2);
  plmtex("t", 1.0, 0.5, 0.5, label_buff); // Title
  plschr(0.0, 1.0/titleScl); // undo scale labels
  // Write out percentile results on figure
  plschr(0.0, resultScl); // scale result text font
  plmtex("lv", -1.0, 0.95, 0.0, "Percentiles");
  sprintf(label_buff, "-3\u03c3: %g", pres[0]);
  plmtex("lv", -1.0, 0.90, 0.0, label_buff);
  sprintf(label_buff, "-2\u03c3: %g", pres[1]);
  plmtex("lv", -1.0, 0.85, 0.0, label_buff);
  sprintf(label_buff, "-1\u03c3: %g", pres[2]);
  plmtex("lv", -1.0, 0.80, 0.0, label_buff);
  sprintf(label_buff, "Median: %g", pres[3]);
  plmtex("lv", -1.0, 0.75, 0.0, label_buff);
  sprintf(label_buff, "+1\u03c3: %g", pres[4]);
  plmtex("lv", -1.0, 0.70, 0.0, label_buff);
  sprintf(label_buff, "+2\u03c3: %g", pres[5]);
  plmtex("lv", -1.0, 0.65, 0.0, label_buff);
  sprintf(label_buff, "+3\u03c3: %g", pres[6]);
  plmtex("lv", -1.0, 0.60, 0.0, label_buff);

  plmtex("rv", -1.0, 0.95, 1.0, "Limits");
  sprintf(label_buff, "3\u03c3 low lim: %g", lres[0]);
  plmtex("rv", -1.0, 0.90, 1.0, label_buff);
  sprintf(label_buff, "2\u03c3 low lim: %g", lres[1]);
  plmtex("rv", -1.0, 0.85, 1.0, label_buff);
  sprintf(label_buff, "1\u03c3 low lim: %g", lres[2]);
  plmtex("rv", -1.0, 0.80, 1.0, label_buff);
  sprintf(label_buff, "1\u03c3 up lim: %g", lres[3]);
  plmtex("rv", -1.0, 0.75, 1.0, label_buff);
  sprintf(label_buff, "2\u03c3 up lim: %g", lres[4]);
  plmtex("rv", -1.0, 0.70, 1.0, label_buff);
  sprintf(label_buff, "3\u03c3 up lim: %g", lres[5]);
  plmtex("rv", -1.0, 0.65, 1.0, label_buff);

  plschr(0.0, 1.0/resultScl); // undo font scale

  // Do plot B chain series of data
  subnr = 1;
  subnc = 1;
  strtr = 2; // start row 0 is bottom
  strtc = 0; // start column 0 is left
  // Set the viewport
  plvpor(vxmin+strtc*delvx, vxmin+(strtc+subnc)*delvx, vymin+strtr*delvy, vymin+(strtr+subnr+0.3)*delvy);

  plwind(0.0, (double) nUse, datamin-ubx, datamax+ubx);
  plschr(0.0, tickScl); // scale ticklabels
  plbox("bcnts", 0.0, 0.0, "bcntsv", 0.0, 0.0);
  plschr(0.0, 1.0/tickScl); // undo ticklabel scale
  plschr(0.0, mrkScl); // resize  for markers
  setclr(fillclr);
  // The 'X' data is just an incrementing counter
  stepInc = gsl_vector_alloc(nUse);
  for (ii=0; ii<nUse; ii++) {
    gsl_vector_set(stepInc, ii, (double)ii);
  }
  // make vectors with stride to subsample chain
  newn = (size_t) ((double) nUse)/((double) stride) + 1;
  xtmp = gsl_vector_alloc(newn);
  ytmp = gsl_vector_alloc(newn);
  j=0;
  for (ii=0; ii<nUse; ii=ii+stride) {
    gsl_vector_set(xtmp, j, gsl_vector_get(stepInc, ii));
    gsl_vector_set(ytmp, j, gsl_vector_get(curDatas, ii));
    j++;
  }
  plstring(newn, gsl_vector_ptr(xtmp, 0), gsl_vector_ptr(ytmp, 0), symbfil); // plot fill markers
  plschr(0.0, 1.0/mrkScl); // undo symbol scaling
  // Labels
  plschr(0.0, labelScl); // scale labels
  setclr(drwclr);
  plsfont(label_family, label_style, label_weight); // font format
  plmtex("b", 2.0, 0.5, 0.5, "Step Number"); // x-axis
  plmtex("l", 2.5, 0.5, 0.5, curname);
  plschr(0.0, 1.0/labelScl); // undo scale labels

  // Do plot C chain autocorrelation
  subnr = 1;
  subnc = 1;
  strtr = 0; // start row 0 is bottom
  strtc = 0; // start column 0 is left
  // Set the viewport
  plvpor(vxmin+strtc*delvx, vxmin+(strtc+subnc)*delvx, vymin+strtr*delvy, vymin+(strtr+subnr+0.3)*delvy);

  plwind(0.0, (double) NLAGS, 0.0, 1.0);
  setclr(drwclr);
  plschr(0.0, tickScl); // scale ticklabels
  plbox("bcnts", 0.0, 0.0, "bcntsv", 0.0, 0.0);
  plschr(0.0, 1.0/tickScl); // undo ticklabel scale

  // The 'X' data is just an incrementing counter
  gsl_vector_free(stepInc);
  gsl_vector_free(ytmp);
  stepInc = gsl_vector_alloc(NLAGS);
  ytmp = gsl_vector_alloc(NLAGS);
  for (ii=0; ii<NLAGS; ii++) {
    gsl_vector_set(stepInc, ii, (double)ii);
  }
  // Calculate autocorrelations at NLAGS lags
  for (i=0; i<NLAGS; i++) {
    gsl_vector_set(ytmp, i, gsl_stats_correlation(gsl_vector_ptr(unviewdata,0), 1, gsl_vector_ptr(unviewdata,i), 1, nUse - i));
  }
  setclr(fillclr);
  plline(NLAGS, gsl_vector_ptr(stepInc, 0), gsl_vector_ptr(ytmp, 0)); // plot fill markers
  // Labels
  plschr(0.0, labelScl); // scale labels
  setclr(drwclr);
  plsfont(label_family, label_style, label_weight); // font format
  plmtex("b", 2.0, 0.5, 0.5, "Lag"); // x-axis
  plmtex("l", 2.5, 0.5, 0.5, "AutoCorr");
  plschr(0.0, 1.0/labelScl); // undo scale labels

  plend();

  gsl_vector_free(stepInc);
  gsl_vector_free(sortdata);
  gsl_vector_free(unviewdata);
  gsl_vector_free(xtmp);
  gsl_vector_free(ytmp);
  gsl_histogram_free(hist_data);
}



int pgmcmc_report(pgmcmc_ioblk *io, pgmcmc_algo_params *pm, pgmcmc_mcmc *mc, pgmcmc_userdata *ud, pgmcmc_userfuncs *uf, char *pnames, char *cnames) {
  int err=0, i, nparm, nfreepar;
  size_t istp, maxstps, clipout=200, curstp, nuse;
  gsl_matrix *pvals, *cvals, *bvals;
  hid_t file_id, dataset_id, dataspace_id;
  hsize_t h5dims[2];
  herr_t h5status;
  char *curname;
  gsl_vector_view curDatas, curLikes, curPriors, curChi2s;
  
  nparm = io->nparm;
  nfreepar = mc->nfreepar;
  maxstps = pm->maxstps;
  curstp = mc->pos;
  nuse = curstp - 1 - clipout;
  if (pm->debugLevel > 0) {
    printf("Report Nparm: %d Nfree: %d Pos: %zu of %zu\n", nparm, nfreepar, curstp, maxstps);
  }
  
  // Allocate parameter storage arrays for reading results
  pvals = gsl_matrix_alloc(maxstps, nfreepar);
  cvals = gsl_matrix_alloc(maxstps, nfreepar);
  bvals = gsl_matrix_alloc(maxstps, 3);
  
  // Open HDF5 file
  file_id = H5Fopen(pm->outputFilename,  H5F_ACC_RDONLY, H5P_DEFAULT);
  // Open dataset
  dataset_id = H5Dopen1(file_id, "/pvals");
  // Read dataset
  h5status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gsl_matrix_ptr(pvals, 0, 0));
  // Close dataset
  h5status = H5Dclose(dataset_id);
  // Repeat for cvals
  // Open dataset
  dataset_id = H5Dopen1(file_id, "/cvals");
  // Read dataset
  h5status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gsl_matrix_ptr(cvals, 0, 0));
  // Close dataset
  h5status = H5Dclose(dataset_id);
  // Do for bvals
  // Open dataset
  dataset_id = H5Dopen1(file_id, "/bvals");
  // Read dataset
  h5status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gsl_matrix_ptr(bvals, 0, 0));
  // Close dataset
  h5status = H5Dclose(dataset_id);
  // Close File
  h5status = H5Fclose(file_id);
  
  // Go through each parameter and make diagnostic calculations and plots
  for (i=0; i<nfreepar; i++) {
    curname = (pnames + i*MAX_NAME_LEN);
    printf("%d %s\n", i, curname);
    curDatas = gsl_matrix_subcolumn(pvals, i, clipout, nuse);
    curLikes = gsl_matrix_subcolumn(bvals, 0, clipout, nuse);
    curPriors = gsl_matrix_subcolumn(bvals, 1, clipout, nuse);
    curChi2s = gsl_matrix_subcolumn(bvals, 2, clipout, nuse);
    parameter_diagnostics(curname, &curDatas.vector, &curLikes.vector, &curPriors.vector, &curChi2s.vector, nuse, 10);
    //curname = (cnames + i*MAX_NAME_LEN);
    //printf("%d %s\n", i, curname);
    //curDatas = gsl_matrix_subcolumn(cvals, i, clipout, nuse);
    //parameter_diagnostics(curname, &curDatas.vector, &curLikes.vector, &curPriors.vector, &curChi2s.vector, nuse, 10);
  }
  
  gsl_matrix_free(pvals);
  gsl_matrix_free(cvals);
  gsl_matrix_free(bvals);
  return err;
}

