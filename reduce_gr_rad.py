#! /usr/bin/env python3

"""
Script for reducing AthenaK radiation-GRMHD simulations.

Usage:
[python3] reduce_gr_rad.py \
    <input_file_base> <output_file_base> \
    <frame_min> <frame_max> <frame_stride> \
    [--r_min <r_in>] [--r_max <r_out>] [--n_r <n_r>] \
    [--lat_max <lat_max>] [--n_th <n_th>]

<input_file_base> should refer to standard AthenaK .bin data dumps that use GR
(Cartesian Kerr-Schild coordinates). Files should have the form
<input_file_base>.{05d}.bin.

The files <output_file_base>.{05d}.npz will be overwritten. They will contain
the remapped and reduced data.

File numbers from <frame_min> to <frame_max>, inclusive with a stride of
<frame_stride>, will be processed.

Options:
  --r_min: minimum radial coordinate r in output grid; default: horizon radius
  --r_max: maximum radial coordinate r in output grid; default: minimum of
      distances along axes to outer boundaries
  --n_r: number of radial zones in output grid; default: 10 cells per decade
  --lat_max: maximum latitude (degrees) grid extends away from midplane;
      default: 90.0
  --n_th: number of polar zones in output grid; default: integer that yields
      approximately square cells given other inputs
  --n_ph: number of azimuthal zones in 3D version of output grid; default:
      integer that yields approximately square cells given other inputs

Reduction is performed as follows: For each cell in the new (phi,theta,r) grid,
the primitives are obtained via trilinear interpolation on the old (z,y,x) grid
to the new cell center. Note that for coarse new grids this subsamples the data,
rather than averaging data in all old cells contained within a new cell. On the
new grid, quantities are transformed to spherical components, and some derived
quantities are calculated. These are all then averaged onto the new (theta,r)
grid, and some further derived quantities are calculated.

Run "reduce_gr_rad.py -h" to see a full description of inputs.
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
  input_file_base = kwargs['input_file_base']
  output_file_base = kwargs['output_file_base']
  frame_min = kwargs['frame_min']
  frame_max = kwargs['frame_max']
  frame_stride = kwargs['frame_stride']
  r_min = kwargs['r_min']
  r_max = kwargs['r_max']
  n_r = kwargs['n_r']
  lat_max = kwargs['lat_max']
  n_th = kwargs['n_th']
  n_ph = kwargs['n_ph']
  dependencies = ('dens',)

  # Go through files
  for frame_n, frame in enumerate(range(frame_min, frame_max + 1, frame_stride)):

    # Calculate file names
    input_file = '{0}.{1:05d}.bin'.format(input_file_base, frame)
    output_file = '{0}.{1:05d}.npz'.format(output_file_base, frame)

    # Read input data
    with open(input_file, 'rb') as f:

      # Get file size
      f.seek(0, 2)
      file_size = f.tell()
      f.seek(0, 0)

      # Read header metadata
      if frame_n == 0:
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
      else:
        for n in range(8):
          next(f)
        line = f.readline().decode('ascii')
        if line[:16] != '  header offset=':
          raise RuntimeError('Could not read header offset.')
        header_offset = int(line[16:])

      # Process header metadata
      if frame_n == 0:
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
      if frame_n == 0:
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
      else:
        f.seek(header_offset, 1)

      # Extract grid limits from input file metadata
      if frame_n == 0:
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
      if frame_n == 0:
        try:
          a = float(input_data['coord']['a'])
        except:
          raise RuntimeError('Unable to find black hole spin in input file.')

      # Prepare lists to hold results
      if frame_n == 0:
        block_lims = []
        data_cks = {}
        for name in variable_names_sorted:
          data_cks[name] = []

      # Go through blocks
      first_time = True if frame_n == 0 else False
      block_n = 0
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
        if frame_n == 0:
          block_lims.append(struct.unpack('=6' + location_format, f.read(6 * location_size)))
        else:
          f.seek(6 * location_size, 1)

        # Read cell data
        cell_data_start = f.tell()
        for ind, name in zip(variable_inds_sorted, variable_names_sorted):
          f.seek(cell_data_start + ind * variable_data_size, 0)
          if frame_n == 0:
            data_cks[name].append(np.array(struct.unpack(block_cell_format, f.read(variable_data_size))).reshape(n_z, n_y, n_x))
          else:
            data_cks[name][block_n] = np.array(struct.unpack(block_cell_format, f.read(variable_data_size))).reshape(n_z, n_y, n_x)
        f.seek((num_variables_base - ind - 1) * variable_data_size, 1)

        # Advance block counter
        block_n += 1

    # Process input data
    if frame_n == 0:
      a2 = a ** 2
      block_lims = np.array(block_lims)
      n_b = block_n
      for name in variable_names_sorted:
        data_cks[name] = np.array(data_cks[name])

    # Adjust parameters
    if frame_n == 0:
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
    if frame_n == 0:
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

    # Construct mapping
    if frame_n == 0:
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

    # Remap data to 3D SKS grid
    if frame_n == 0:
      data_3d = {}
      data_3d['rho'] = np.empty((n_ph, n_th, n_r))
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
          data_3d['rho'][ind_ph,ind_th,ind_r] = (1.0 - weight_z) * ((1.0 - weight_y) * ((1.0 - weight_x) * val_mmm + weight_x * val_mmp) + weight_y * ((1.0 - weight_x) * val_mpm + weight_x * val_mpp)) + weight_z * ((1.0 - weight_y) * ((1.0 - weight_x) * val_pmm + weight_x * val_pmp) + weight_y * ((1.0 - weight_x) * val_ppm + weight_x * val_ppp))

    # TODO: Convert data to SKS components

    # TODO: Calculate pre-averaging derived quantities

    # Average quantities in azimuth
    if frame_n == 0:
      data_2d = {}
    data_2d['rho'] = np.mean(data_3d['rho'], axis=0)

    # TODO: Calculate post-averaging derived quantities

    # Save results
    data_out = {}
    data_out['rf'] = rf
    data_out['r'] = r
    data_out['thf'] = thf
    data_out['th'] = th
    data_out['phf'] = phf
    data_out['ph'] = ph
    data_out['rho'] = data_2d['rho']
    np.savez(output_file, **data_out)

# Process inputs and execute main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file_base', help='base name of AthenaK .bin files to reduce')
  parser.add_argument('output_file_base', help='base name of .npz files to write')
  parser.add_argument('frame_min', type=int, help='initial file number to process')
  parser.add_argument('frame_max', type=int, help='final file number to process')
  parser.add_argument('frame_stride', type=int, help='stride in file numbers to process')
  parser.add_argument('--r_min', type=float, help='minimum radial coordinate r in output grid')
  parser.add_argument('--r_max', type=float, help='maximum radial coordinate r in output grid')
  parser.add_argument('--n_r', type=int, help='number of radial zones in output grid')
  parser.add_argument('--lat_max', type=float, help='maximum latitude (degrees) grid extends away from midplane')
  parser.add_argument('--n_th', type=int, help='number of polar zones in output grid')
  parser.add_argument('--n_ph', type=int, help='number of azimuthal zones in 3D version of output grid')
  args = parser.parse_args()
  main(**vars(args))
