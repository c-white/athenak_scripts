#! /usr/bin/env python3

"""
Script for calculating AthenaK data remapping.

Usage:
[python3] calculate_cks_3d_to_sks_2d_mapping.py \
    <input_file> <output_file> \
    [--r_min <r_in>] [--r_max <r_out>] [--n_r <n_r>] \
    [--lat_max <lat_max>] [--n_th <n_th>]

<input_file> can be any standard AthenaK .bin data dump that uses GR (Cartesian
Kerr-Schild coordinates).

<output_file> will be overwritten. It will be a .npz file containing the
information needed to map any .bin file with the same layout as <input_file>
into a 2D spherical Kerr-Schild r-theta grid via azimuthal averaging.

Options:
  --r_min: minimum radial coordinate r in output grid; default: horizon radius
  --r_max: maximum radial coordinate r in output grid; default: minimum of
      distances along axes to outer boundaries
  --n_r: number of radial zones in output grid; default: 10 cells per decade
  --lat_max: maximum latitude (degrees) grid extends away from midplane;
      default: 90.0
  --n_th: number of polar zones in output grid; default: integer that yields
      approximately square cells given other inputs
  --n_ph: number of azimuthal zones in 3D version of output grid, used only if
      coarse_out is not set; default: integer that yields approximately square
      cells given other inputs
  --coarse_out: flag indicating mapping can be optimized for output grid that is
      coarse relative to input grid

Run "calculate_cks_3d_to_sks_2d.py -h" to see a full description of inputs.

By default, mapping is the following: for each cell in the new grid, a list of
n_ph old-grid indices and trilinear interpolation coefficients for obtaining
values from the old grid. If coarse_out is set, mapping is the following: for
each cell in the old grid, indices and weights for adding values to new grid.
The latter is dangerous in that it can lead to unfilled cells on new grids that
are too fine. However, it may be cheaper. The default method is safe though
possibly expensive, and it reduces to subsampling if used on coarse new grids.
"""

# Python standard modules
import argparse
import struct

# Numerical modules
import numpy as np

# Main function
def main(**kwargs):

  # Parameters
  cells_per_decade = 10.0
  r_square = 10.0
  input_file = kwargs['input_file']
  output_file = kwargs['output_file']
  r_min = kwargs['r_min']
  r_max = kwargs['r_max']
  n_r = kwargs['n_r']
  lat_max = kwargs['lat_max']
  n_th = kwargs['n_th']
  n_ph = kwargs['n_ph']
  coarse_out = kwargs['coarse_out']

  # Read data
  with open(kwargs['input_file'], 'rb') as f:

    # Get file size
    f.seek(0, 2)
    file_size = f.tell()
    f.seek(0, 0)

    # Read header metadata
    line = f.readline().decode('ascii')
    if line != 'Athena binary output version=1.1\n':
      raise RuntimeError('Unrecognized data file format.')
    next(f)
    next(f)
    next(f)
    line = f.readline().decode('ascii')
    if line[:19] != '  size of location=':
      raise RuntimeError('Could not read location size.')
    location_size = int(line[19:])
    line = f.readline().decode('ascii')
    if line[:19] != '  size of variable=':
      raise RuntimeError('Could not read variable size.')
    variable_size = int(line[19:])
    next(f)
    line = f.readline().decode('ascii')
    if line[:12] != '  variables:':
      raise RuntimeError('Could not read variable names.')
    variable_names_base = line[12:].split()
    line = f.readline().decode('ascii')
    if line[:16] != '  header offset=':
      raise RuntimeError('Could not read header offset.')
    header_offset = int(line[16:])

    # Process header metadata
    if location_size not in (4, 8):
      raise RuntimeError('Only 4- and 8-byte integer types supported for location data.')
    location_format = 'f' if location_size == 4 else 'd'
    if variable_size not in (4, 8):
      raise RuntimeError('Only 4- and 8-byte integer types supported for cell data.')
    num_variables_base = len(variable_names_base)

    # Read input file metadata
    input_data = {}
    start_of_data = f.tell() + header_offset
    while f.tell() < start_of_data:
      line = f.readline().decode('ascii')
      if line[0] == '#':
        continue
      if line[0] == '<':
        section_name = line[1:-2]
        input_data[section_name] = {}
        continue
      key, val = line.split('=', 1)
      input_data[section_name][key.strip()] = val.split('#', 1)[0].strip()

    # Extract grid limits from input file metadata
    try:
      x1_min = float(input_data['mesh']['x1min'])
      x1_max = float(input_data['mesh']['x1max'])
      x2_min = float(input_data['mesh']['x2min'])
      x2_max = float(input_data['mesh']['x2max'])
      x3_min = float(input_data['mesh']['x3min'])
      x3_max = float(input_data['mesh']['x3max'])
    except:
      raise RuntimeError('Unable to find grid limits in input file.')

    # Extract black hole spin from input file metadata
    try:
      a = float(input_data['coord']['a'])
    except:
      raise RuntimeError('Unable to find black hole spin in input file.')

    # Prepare list to hold results
    block_lims = []

    # Go through blocks
    first_time = True
    while f.tell() < file_size:

      # Read and process grid structure data
      if first_time:
        block_indices = struct.unpack('@6i', f.read(24))
        n_x = block_indices[1] - block_indices[0] + 1
        n_y = block_indices[3] - block_indices[2] + 1
        n_z = block_indices[5] - block_indices[4] + 1
        cells_per_block = n_z * n_y * n_x
        variable_data_size = cells_per_block * variable_size
        first_time = False
      else:
        f.seek(24, 1)
      f.seek(16, 1)

      # Read coordinate data
      block_lims.append(struct.unpack('=6' + location_format, f.read(6 * location_size)))

      # Skip cell data
      f.seek(num_variables_base * variable_data_size, 1)

  # Process input grid data
  a2 = a ** 2
  block_lims = np.array(block_lims)
  n_b = block_lims.shape[0]

  # Adjust parameters
  if r_min is None:
    r_min = 1.0 + (1.0 - a2) ** 0.5
  if r_max is None:
    r_max = min(-x1_min, x1_max, -x2_min, x2_max, -x3_min, x3_max)
  if n_r is None:
    n_r = int(round(np.log10(r_max / r_min) * cells_per_decade))
  if lat_max is None:
    th_min = 0.0
    th_max = np.pi
  else:
    th_min = np.pi/2.0 - lat_max * np.pi/180.0
    th_max = np.pi/2.0 + lat_max * np.pi/180.0
  if n_th is None:
    n_th = int(round((1.0 + 2.0 / r_square) ** -0.5 * cells_per_decade * np.log10(np.e) * (th_max - th_min)))
  ph_min = 0.0
  ph_max = 2.0*np.pi
  if n_ph is None:
    n_ph = int(round(((1.0 + (1.0 + 2.0 / r_square) * a2 / r_square ** 2) / (1.0 + 2.0 / r_square)) ** 0.5 * cells_per_decade * np.log10(np.e) * 2.0*np.pi))

  # Construct new grid
  lrf = np.linspace(np.log(r_min), np.log(r_max), n_r + 1)
  lr = 0.5 * (lrf[:-1] + lrf[1:])
  rf = np.exp(lrf)
  r = np.exp(lr)
  thf = np.linspace(th_min, th_max, n_th + 1)
  th = 0.5 * (thf[:-1] + thf[1:])
  sth = np.sin(th)
  cth = np.cos(th)
  phf = np.linspace(ph_min, ph_max, n_ph + 1)
  ph = 0.5 * (phf[:-1] + phf[1:])
  sph = np.sin(ph)
  cph = np.cos(ph)

  # Construct mapping for fine new grid
  if not coarse_out:
    inds = np.empty((4, n_ph, n_th, n_r), dtype=int)
    weights = np.empty((3, n_ph, n_th, n_r))
    for ind_ph in range(n_ph):
      for ind_th in range(n_th):
        for ind_r in range(n_r):
          x_val = sth[ind_th] * (r[ind_r] * cph[ind_ph] - a * sph[ind_ph])
          y_val = sth[ind_th] * (r[ind_r] * sph[ind_ph] + a * cph[ind_ph])
          z_val = r[ind_r] * cth[ind_th]
          x_inds = (x_val >= block_lims[:,0]) & (x_val < block_lims[:,1])
          y_inds = (y_val >= block_lims[:,2]) & (y_val < block_lims[:,3])
          z_inds = (z_val >= block_lims[:,4]) & (z_val < block_lims[:,5])
          ind_b = np.where(x_inds & y_inds & z_inds)[0][0]
          inds[0,ind_ph,ind_th,ind_r] = ind_b
          ind_frac_x = (x_val - block_lims[ind_b,0]) / (block_lims[ind_b,1] - block_lims[ind_b,0]) * n_x - 0.5
          ind_frac_y = (y_val - block_lims[ind_b,2]) / (block_lims[ind_b,3] - block_lims[ind_b,2]) * n_y - 0.5
          ind_frac_z = (z_val - block_lims[ind_b,4]) / (block_lims[ind_b,5] - block_lims[ind_b,4]) * n_z - 0.5
          ind_x = min(int(ind_frac_x), n_x - 2)
          ind_y = min(int(ind_frac_y), n_y - 2)
          ind_z = min(int(ind_frac_z), n_z - 2)
          inds[1,ind_ph,ind_th,ind_r] = ind_z
          inds[2,ind_ph,ind_th,ind_r] = ind_y
          inds[3,ind_ph,ind_th,ind_r] = ind_x
          weights[0,ind_ph,ind_th,ind_r] = ind_frac_z - ind_z
          weights[1,ind_ph,ind_th,ind_r] = ind_frac_y - ind_y
          weights[2,ind_ph,ind_th,ind_r] = ind_frac_x - ind_x

  # Construct mapping for coarse new grid
  if coarse_out:
    inds = np.empty((2, n_b, n_z, n_y, n_x), dtype=int)
    weights = np.empty((n_b, n_z, n_y, n_x))
    x = np.empty((n_b, n_x))
    y = np.empty((n_b, n_y))
    z = np.empty((n_b, n_z))
    for ind_b in range(n_b):
      xf = np.linspace(block_lims[ind_b,0], block_lims[ind_b,1], n_x + 1)
      yf = np.linspace(block_lims[ind_b,2], block_lims[ind_b,3], n_y + 1)
      zf = np.linspace(block_lims[ind_b,4], block_lims[ind_b,5], n_z + 1)
      x[ind_b,:] = 0.5 * (xf[:-1] + xf[1:])
      y[ind_b,:] = 0.5 * (yf[:-1] + yf[1:])
      z[ind_b,:] = 0.5 * (zf[:-1] + zf[1:])
    rr2_vals = x[:,None,None,:] ** 2 + y[:,None,:,None] ** 2 + z[:,:,None,None] ** 2
    r2_vals = 0.5 * (rr2_vals - a2 + np.hypot(rr2_vals - a2, 2.0 * a * z[:,:,None,None]))
    r_vals = np.sqrt(r2_vals)
    th_vals = np.arccos(z[:,:,None,None] / r_vals)
    inds[0,:,:,:,:] = ((th_vals - thf[0]) / (thf[-1] - thf[0]) * n_th).astype(int)
    inds[1,:,:,:,:] = ((np.log(r_vals) - lrf[0]) / (lrf[-1] - lrf[0]) * n_r).astype(int)
    for ind_b in range(n_b):
      delta_x = block_lims[ind_b,1] - block_lims[ind_b,0]
      delta_y = block_lims[ind_b,3] - block_lims[ind_b,2]
      delta_z = block_lims[ind_b,5] - block_lims[ind_b,4]
      weights[ind_b,:,:,:] = delta_x * delta_y * delta_z
    weights = np.where((inds[0,:,:,:,:] < 0) | (inds[0,:,:,:,:] > n_th - 1) | (inds[1,:,:,:,:] < 0) | (inds[1,:,:,:,:] > n_r - 1), 0.0, weights)
    for ind_th in range(n_th):
      for ind_r in range(n_r):
        inds_match = (inds[0,:,:,:,:] == ind_th) & (inds[1,:,:,:,:] == ind_r)
        weight = np.sum(np.where(inds_match, weights, 0.0))
        weights[inds_match] /= weight

  # Save results
  data = {}
  data['rf'] = rf
  data['r'] = r
  data['thf'] = thf
  data['th'] = th
  data['phf'] = phf
  data['ph'] = ph
  data['inds'] = inds
  data['weights'] = weights
  np.savez(output_file, **data)

# Process inputs and execute main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file', help='name of AthenaK .bin file containing desired input grid')
  parser.add_argument('output_file', help='name of .npz file to write')
  parser.add_argument('--r_min', type=float, help='minimum radial coordinate r in output grid')
  parser.add_argument('--r_max', type=float, help='maximum radial coordinate r in output grid')
  parser.add_argument('--n_r', type=int, help='number of radial zones in output grid')
  parser.add_argument('--lat_max', type=float, help='maximum latitude (degrees) grid extends away from midplane')
  parser.add_argument('--n_th', type=int, help='number of polar zones in output grid')
  parser.add_argument('--n_ph', type=int, help='number of azimuthal zones in 3D version of output grid, used only if coarse_out is not set')
  parser.add_argument('--coarse_out', action='store_true', help='flag indicating mapping can be optimized for output grid that is coarse relative to input grid')
  args = parser.parse_args()
  main(**vars(args))
