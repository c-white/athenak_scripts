#! /usr/bin/env python3

"""
Script for plotting reduced AthenaK data.

Usage:
[python3] plot_reduced_gr_rad.py <data_file> <variable> <output_file> [options]

<data_file> should be the output of reduce_gr_rad.py.

<variable> should be any of the cell quantities saved by
reduce_gr_rad.py.

<output_file> will be overwritten with a plot of the desired quantity. If
<output_file> is simply "show", the script will open a live Matplotlib viewing
window.

Optional inputs include:
  --r_max: maximum radial coordinate to plot
  -c: colormap recognized by Matplotlib
  -n: colormap normalization (e.g., "-n log") if not linear
  --vmin, --vmax: limits of colorbar if not the full range of data
  --notex: flag to disable Latex typesetting of labels
  --ergosphere: flag for outlining boundary of ergosphere in GR simulation
  --ergosphere_color, horizon_mask_color: color choices

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
  import matplotlib.colors as colors
  import matplotlib.patches as patches
  import matplotlib.pyplot as plt

  # Plotting parameters
  ergosphere_num_points = 129
  ergosphere_line_style = '-'
  ergosphere_line_width = 1.0
  x1_labelpad = 2.0
  x2_labelpad = 2.0
  dpi = 300

  # Read data
  data = np.load(kwargs['data_file'])
  try:
    quantity = data[kwargs['variable']]
  except KeyError:
    raise RuntimeError('{0} not in file.'.format(kwargs['variable']))
  a = data['a']
  rf = data['rf']
  thf = data['thf']

  # Process data
  quantity_right = quantity
  quantity_left = quantity[::-1,:]
  thf_right = thf
  thf_left = 2.0*np.pi - thf[::-1]
  xc_right = rf[None,:] * np.sin(thf_right[:,None])
  xc_left = rf[None,:] * np.sin(thf_left[:,None])
  zc_right = rf[None,:] * np.cos(thf_right[:,None])
  zc_left = rf[None,:] * np.cos(thf_left[:,None])

  # Calculate colors
  if kwargs['vmin'] is None:
    vmin = np.nanmin(quantity)
  else:
    vmin = kwargs['vmin']
  if kwargs['vmax'] is None:
    vmax = np.nanmax(quantity)
  else:
    vmax = kwargs['vmax']

  # Choose colormap norm
  if kwargs['norm'] == 'linear':
    norm = colors.Normalize(vmin, vmax)
    vmin = None
    vmax = None
  elif kwargs['norm'] == 'log':
    norm = colors.LogNorm(vmin, vmax)
    vmin = None
    vmax = None
  else:
    norm = kwargs['norm']

  # Prepare figure
  plt.figure()

  # Plot data
  plt.pcolormesh(xc_right, zc_right, quantity_right, cmap=kwargs['cmap'], norm=norm, vmin=vmin, vmax=vmax)
  plt.pcolormesh(xc_left, zc_left, quantity_left, cmap=kwargs['cmap'], norm=norm, vmin=vmin, vmax=vmax)

  # Make colorbar
  plt.colorbar()

  # Mask horizon
  r_hor = 1.0 + (1.0 - a ** 2) ** 0.5
  full_width = 2.0 * (r_hor ** 2 + a ** 2) ** 0.5
  full_height = 2.0 * ((r_hor ** 2 + a ** 2) / (1.0 + a ** 2 / r_hor ** 2)) ** 0.5
  horizon_mask = \
      patches.Circle((0.0, 0.0), r_hor, facecolor=kwargs['horizon_mask_color'], edgecolor='none')
  plt.gca().add_artist(horizon_mask)

  # Mark ergosphere
  if kwargs['ergosphere']:
    th_plot = np.linspace(0.0, np.pi, ergosphere_num_points)
    th_plot = np.concatenate((th_plot, 2.0*np.pi - th_plot[-2::-1]))
    sth_plot = np.sin(th_plot)
    cth_plot = np.cos(th_plot)
    r_plot = 1.0 + np.sqrt(1.0 - a ** 2 * cth_plot ** 2)
    x_plot = r_plot * sth_plot
    y_plot = r_plot * cth_plot
    plt.plot(x_plot, y_plot, linestyle=ergosphere_line_style, linewidth=ergosphere_line_width, \
        color=kwargs['ergosphere_color'])

  # Adjust axes
  r_max = rf[-1]
  if kwargs['r_max'] is not None:
    r_max = kwargs['r_max']
  plt.xlim((-r_max, r_max))
  plt.ylim((-r_max, r_max))
  plt.xlabel(r'$r \sin\theta$', labelpad=x1_labelpad)
  plt.ylabel(r'$r \cos\theta$', labelpad=x2_labelpad)

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
  parser.add_argument('output_file', \
      help='name of output to be (over)written; use "show" to show interactive plot instead')
  parser.add_argument('--r_max', type=float, help='maximum radial coordinate to plot')
  parser.add_argument('-c', '--cmap', help='name of Matplotlib colormap to use')
  parser.add_argument('-n', '--norm', help='name of Matplotlib norm to use')
  parser.add_argument('--vmin', type=float, help='colormap minimum')
  parser.add_argument('--vmax', type=float, help='colormap maximum')
  parser.add_argument('--notex', action='store_true', \
      help='flag indicating LaTeX integration is not to be used')
  parser.add_argument('--ergosphere', action='store_true', \
      help='flag indicating black hole ergosphere should be marked')
  parser.add_argument('--ergosphere_color', default='gray', \
      help='color string for ergosphere marker')
  parser.add_argument('--horizon_mask_color', default='k', \
      help='color string for event horizon mask')
  args = parser.parse_args()
  main(**vars(args))
