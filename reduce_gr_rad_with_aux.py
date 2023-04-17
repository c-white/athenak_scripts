#! /usr/bin/env python3

"""
Script for reducing AthenaK radiation-GRMHD simulations.

Usage:
[python3] reduce_gr_rad_with_aux.py <aux_file> <input_file> <output_file>

<aux_file> should be the output of calculate_cks_3d_to_sks_2d_mapping.py.

<input_file> can be any AthenaK .bin data dump compatible with the input to
calculate_cks_3d_to_sks_2d_mapping.py.

<output_file> will be overwritten. It will be a .npz file with a (theta, r)
spherical Kerr-Schild grid and various quantities defined on that grid.

Run "reduce_gr_rad_with_aux.py -h" to see a full description of inputs.
"""

# Python standard modules
import argparse
import struct

# Numerical modules
import numpy as np

# Main function
def main(**kwargs):

  # Parameters
  aux_file = kwargs['aux_file']
  input_file = kwargs['input_file']
  output_file = kwargs['output_file']
  dependencies = ('dens',)

  # Read auxiliary data
  data_aux = np.load(aux_file)

  # Process auxiliary data
  r = data_aux['r']
  n_r = len(r)
  th = data_aux['th']
  n_th = len(th)
  n_ph = len(data_aux['ph'])
  inds = data_aux['inds']
  weights = data_aux['weights']

  # Read input data
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
    variable_format = 'f' if variable_size == 4 else 'd'
    num_variables_base = len(variable_names_base)
    variable_names = []
    variable_inds = []
    for dependency in dependencies:
      if dependency not in variable_names_base:
        raise RuntimeError('Required variable {0} not found.'.format(dependency))
      variable_names.append(dependency)
      variable_ind = 0
      while variable_names_base[variable_ind] != dependency:
        variable_ind += 1
      variable_inds.append(variable_ind)
    variable_names_sorted = [name for _, name in sorted(zip(variable_inds, variable_names))]
    variable_inds_sorted = [ind for ind, _ in sorted(zip(variable_inds, variable_names))]

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

    # Extract black hole spin from input file metadata
    try:
      a = float(input_data['coord']['a'])
    except:
      raise RuntimeError('Unable to find black hole spin in input file.')

    # Prepare lists to hold results
    block_lims = []
    data_cks = {}
    for name in variable_names_sorted:
      data_cks[name] = []

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
        block_cell_format = '=' + str(cells_per_block) + variable_format
        variable_data_size = cells_per_block * variable_size
        first_time = False
      else:
        f.seek(24, 1)
      f.seek(16, 1)

      # Read coordinate data
      block_lims.append(struct.unpack('=6' + location_format, f.read(6 * location_size)))

      # Read cell data
      cell_data_start = f.tell()
      for ind, name in zip(variable_inds_sorted, variable_names_sorted):
        f.seek(cell_data_start + ind * variable_data_size, 0)
        data_cks[name].append(np.array(struct.unpack(block_cell_format, f.read(variable_data_size))).reshape(n_z, n_y, n_x))
      f.seek((num_variables_base - ind - 1) * variable_data_size, 1)

  # Process input data
  block_lims = np.array(block_lims)
  n_b = block_lims.shape[0]
  for name in variable_names_sorted:
    data_cks[name] = np.array(data_cks[name])

  # TODO: Convert to SKS components

  # TODO: Calculate pre-averaging derived quantities

  # Convert data to fine SKS grid
  if len(inds.shape) == 4:
    data_sks = {}
    data_sks['rho'] = np.zeros((n_th, n_r))
    for ind_ph in range(n_ph):
      for ind_th in range(n_th):
        for ind_r in range(n_r):
          ind_b = inds[0,ind_ph,ind_th,ind_r]
          ind_z = inds[1,ind_ph,ind_th,ind_r]
          ind_y = inds[2,ind_ph,ind_th,ind_r]
          ind_x = inds[3,ind_ph,ind_th,ind_r]
          weight_z = weights[0,ind_ph,ind_th,ind_r]
          weight_y = weights[1,ind_ph,ind_th,ind_r]
          weight_x = weights[2,ind_ph,ind_th,ind_r]
          val_mmm = data_cks['dens'][ind_b,ind_z,ind_y,ind_x]
          val_mmp = data_cks['dens'][ind_b,ind_z,ind_y,ind_x+1]
          val_mpm = data_cks['dens'][ind_b,ind_z,ind_y+1,ind_x]
          val_mpp = data_cks['dens'][ind_b,ind_z,ind_y+1,ind_x+1]
          val_pmm = data_cks['dens'][ind_b,ind_z+1,ind_y,ind_x]
          val_pmp = data_cks['dens'][ind_b,ind_z+1,ind_y,ind_x+1]
          val_ppm = data_cks['dens'][ind_b,ind_z+1,ind_y+1,ind_x]
          val_ppp = data_cks['dens'][ind_b,ind_z+1,ind_y+1,ind_x+1]
          data_sks['rho'][ind_th,ind_r] += (1.0 - weight_z) * ((1.0 - weight_y) * ((1.0 - weight_x) * val_mmm + weight_x * val_mmp) + weight_y * ((1.0 - weight_x) * val_mpm + weight_x * val_mpp)) + weight_z * ((1.0 - weight_y) * ((1.0 - weight_x) * val_pmm + weight_x * val_pmp) + weight_y * ((1.0 - weight_x) * val_ppm + weight_x * val_ppp))
    data_sks['rho'] /= n_ph

  # Convert data to coarse SKS grid
  if len(inds.shape) == 5:
    n_b = inds.shape[1]
    n_z = inds.shape[2]
    n_y = inds.shape[3]
    n_x = inds.shape[4]
    data_sks = {}
    data_sks['rho'] = np.zeros((n_th, n_r))
    for ind_b in range(n_b):
      for ind_z in range(n_z):
        for ind_y in range(n_y):
          for ind_x in range(n_x):
            ind_th = inds[0,ind_b,ind_z,ind_y,ind_x]
            ind_r = inds[1,ind_b,ind_z,ind_y,ind_x]
            weight = weights[ind_b,ind_z,ind_y,ind_x]
            if weight > 0.0:
              val = data_cks['dens'][ind_b,ind_z,ind_y,ind_x]
              data_sks['rho'][ind_th,ind_r] += weight * val

  # TODO: Calculate post-averaging derived quantities

# Process inputs and execute main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('aux_file', help='auxiliary file produced by calculate_cks_3d_to_sks_2d_mapping.py')
  parser.add_argument('input_file', help='name of AthenaK .bin file with data to reduce')
  parser.add_argument('output_file', help='name of .npz file to write')
  args = parser.parse_args()
  main(**vars(args))
