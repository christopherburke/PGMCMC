/* Routine to generate fake data that follows a linear line
 with Gaussian noise
*/
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <plplot/plplot.h>
#include <math.h>

int main(int argc, char const *argv[]) {
  // Parameters for routine
  size_t nPts = 50; // Number of points to simulate
  double slope = 1.0e-4/100; // slope of line
  double intercept = 20.0; // intercept of line
  double yerr = 3.0e-4; // Gaussian error on data points
  double xmin = 1000.0; // Minimum x-axis data
  double xmax = 1500.0; // Maximum x-axis data
  char *outFile = "gen_linear_data.txt";

  //Plotting formatting
  const char *symbout = "\xE2\x97\x8B"; // UTF-8 circle
  //const char *symbout = "\xE2\xAD\x98"; // heavy circle
  const char *symbfil = "\xE2\x97\x8F"; // black circle
  // PDF output
  //char *plot_dev = "pdfcairo"; // output device
  //char *plot_output_filename = "gen_linear_data.pdf";
  // PNG output
  char *plot_dev = "pngcairo";
  char *plot_output_filename = "gen_linear_data.png";
  double dpi = 400.0; double xWidthIn = 7.0; double yWidthIn = 5.0; // dpi, x axis width in inches

  char *xLabel = "X";
  char *yLabel = "Y";
  PLINT label_family = PL_FCI_SANS; PLINT label_style=PL_FCI_UPRIGHT; PLINT label_weight=PL_FCI_BOLD;
  char *pltTitle = "Linear Model With Gaussian Noise";
  PLINT title_family = PL_FCI_SANS; PLINT title_style=PL_FCI_UPRIGHT; PLINT title_weight=PL_FCI_BOLD;

  double labelScl = 1.5;
  double titleScl = 1.5;
  double mrkScl = 1.5;
  double boxWidth = 3.0;
  double edgebuff_fracx = 0.1; // fraction of data width to add to edges of axis as buffer
  double edgebuff_fracy = 0.1;
  unsigned char bkgclr[] = {0xff, 0xff, 0xff}; // White background
  unsigned char drwclr[] = {0, 0, 0}; // black color
  unsigned char fillclr[] = {61, 156, 235}; // light blue

  // Random number generation variables
  unsigned long int ranseed = 349404040; // Fixed random seed
  gsl_rng *ranobj; // Generator object

  gsl_vector *xData, *yData;
  size_t i;
  double x,y,data_xmin,data_xmax,data_ymin,data_ymax, ubx, uby;
  FILE *fp;

  // Allocate arrays for data
  xData = gsl_vector_alloc(nPts);
  yData = gsl_vector_alloc(nPts);
  // Instantiate random generator struct
  ranobj = gsl_rng_alloc(gsl_rng_mt19937); // gsl_rng_mt19937 == Mersenne twister
  gsl_rng_set(ranobj, ranseed); // Seed Generator
  // open FILe
  fp = fopen(outFile,"w");

  // make x Data random uniform betwee xmin and xmax
  // Y data is model plus gaussian error
  for (i=0; i<nPts; i++)
  {
    x = gsl_ran_flat(ranobj, xmin, xmax);
    y =  gsl_ran_gaussian(ranobj, yerr);
    y = y + intercept + slope * x; // line model with gaussian noise
    gsl_vector_set(xData, i, x);
    gsl_vector_set(yData, i, y);
    fprintf(fp,"%g %g\n", x, y);
  }

  // Now lets try to make a figure of the random data using plplot
  plscolbg(bkgclr[0], bkgclr[1], bkgclr[2]); // set background color
  plsdev(plot_dev); // Set output device
  plsfnam(plot_output_filename); // plot output file name
  // ***PNG SET DPI***
  plspage(dpi, dpi, lround(dpi*xWidthIn), lround(dpi*yWidthIn), 0, 0);
  plinit(); // initialize plot
  plscol0(15, drwclr[0], drwclr[1], drwclr[2]); // colormap entry 15 to drwclor
  plcol0(15); // switch to drwclr for drawing axis
  // Get the min and max of the x & y data vectors
  gsl_vector_minmax(xData, &data_xmin, &data_xmax);
  gsl_vector_minmax(yData, &data_ymin, &data_ymax);
  ubx = fabs((data_xmax - data_xmin) * edgebuff_fracx);
  uby = fabs((data_ymax - data_ymin) * edgebuff_fracy);
  // define data coordinates of plot box
  plsfont(label_family, label_style, label_weight); // font format
  plwidth(boxWidth); // set box line width
  plenv(data_xmin - ubx, data_xmax + ubx, data_ymin - uby, data_ymax + uby, 0, 0);
  // Labels
  plschr(0.0, labelScl); // scale labels
  plsfont(label_family, label_style, label_weight); // font format
  plmtex("b", 2.0, 0.5, 0.5, xLabel); // x-axis
  plmtex("l", 2.5, 0.5, 0.5, yLabel);
  plschr(0.0, 1.0/labelScl); // undo scale labels

  // Title
  plschr(0.0, titleScl); // scale title
  plsfont(title_family, title_style, title_weight); // font format
  plmtex("t", 1.0, 0.5, 0.5, pltTitle); // Title
  plschr(0.0, 1.0/titleScl); // undo scale labels


  //plot data
  plschr(0.0, mrkScl); // resize markers
  plscol0(15, fillclr[0], fillclr[1], fillclr[2]); // fill color
  plcol0(15);
  plstring(nPts, gsl_vector_ptr(xData,0), gsl_vector_ptr(yData,0), symbfil); // plot fill markers
  plscol0(15, drwclr[0], drwclr[1], drwclr[2]); // draw color
  plcol0(15);
  plstring(nPts, gsl_vector_ptr(xData,0), gsl_vector_ptr(yData,0), symbout); // plot outline markers
  plschr(0.0, 1.0/mrkScl); // undo symbol scaling
  plend();

  gsl_vector_free(xData);
  gsl_vector_free(yData);
  gsl_rng_free(ranobj);
  return 0;
}
