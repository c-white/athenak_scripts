#! /usr/bin/env python3

"""
Script for plotting reduced AthenaK data.

Usage:
[python3] plot_reduced_gr_rad.py <data_file> <variable> <output_file>

<data_file> should be the output of reduce_gr_rad.py.

<variable> should be any of the cell quantities saved by
reduce_gr_rad.py.

<output_file> will be overwritten with a plot of the desired quantity. If
<output_file> is simply "show", the script will open a live Matplotlib viewing
window.

Run "plot_reduced_gr_rad.py -h" to see a full description of inputs.
"""

# Python standard modules
import argparse

# Numerical modules
import numpy as np

# Load plotting modules
import matplotlib

# Main function
def main(**kwargs):

  # Load additional plotting modules
  if kwargs['output_file'] != 'show':
    matplotlib.use('agg')
  if not kwargs['notex']:
    matplotlib.rc('text', usetex=True)
  import matplotlib.pyplot as plt

  # Plotting parameters
  dpi = 300

  # Load data
  data = np.load(kwargs['data_file'])
  try:
    variable = data[kwargs['variable']]
  except KeyError:
    raise RuntimeError('{0} not in file.'.format(kwargs['variable']))
  variable = np.vstack((variable, variable[::-1,:]))

  # Calculate grid
  rf = data['rf']
  thf = data['thf']
  thf_ext = np.concatenate((thf, 2.0*np.pi - thf[-2::-1]))
  xc = rf[None,:] * np.sin(thf_ext[:,None])
  zc = rf[None,:] * np.cos(thf_ext[:,None])

  # Prepare figure
  plt.figure()

  # Plot data
  plt.pcolormesh(xc, zc, variable)

  # Make colorbar
  plt.colorbar()

  # Adjust layout
  plt.tight_layout()

  # Save or display figure
  if kwargs['output_file'] != 'show':
    plt.savefig(kwargs['output_file'], dpi=dpi)
  else:
    plt.show()

# Parse inputs and execute main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data_file', help='name of input file, possibly including path')
  parser.add_argument('variable', help='name of variable to be plotted')
  parser.add_argument('output_file', help='name of output to be (over)written; use "show" to show interactive plot instead')
  parser.add_argument('--notex', action='store_true', help='flag indicating LaTeX integration is not to be used')
  args = parser.parse_args()
  main(**vars(args))
