#! /usr/bin/env python3

"""
Script for converting CKS AthenaK GRMHD data to SKS NumPy format.

Usage:
[python3] cks_to_sks.py \
    <input_file_base> <output_file_base> \
    <frame_min> <frame_max> <frame_stride> \
    [--r_min <r_in>] [--r_max <r_out>] [--n_r <n_r>] \
    [--lat_max <lat_max>] [--n_th <n_th>] \
    [--n_ph <n_ph>] \
    [--no_interp] [--no_rad] \
    [--mpi]

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
  --n_r: number of radial zones in output grid; default: 32 cells per decade
  --lat_max: maximum latitude (degrees) grid extends away from midplane;
      default: 90.0
  --n_th: number of polar zones in output grid; default: integer that yields
      approximately square cells given other inputs
  --n_ph: number of azimuthal zones in 3D version of output grid; default:
      integer that yields approximately square cells given other inputs
  --no_interp: flag indicating remapping to be done with nearest neighbors
      rather than interpolation
  --no_rad: flag indicating radiation quantities should be ignored
  --mpi: flag indicating MPI should be used to distribute work

Mapping is performed as follows: For each cell in the new (phi,theta,r) grid,
the primitives are obtained via trilinear interpolation on the old (z,y,x) grid
to the new cell center. Note that for coarse new grids this subsamples the data,
rather than averaging data in all old cells contained within a new cell. On the
new grid, quantities are transformed to spherical components.
"""

# Python standard modules
import argparse
import struct

# Numerical modules
import numpy as np

# Main function
def main(**kwargs):

  # Parameters - fixed
  cells_per_decade_default = 32.0
  r_square = 10.0

  # Parameters - inputs
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
  interp = not kwargs['no_interp']
  include_rad = not kwargs['no_rad']
  use_mpi = kwargs['mpi']

  # Parameters - quantities to extract
  quantities_to_extract = ['dens', 'eint', 'velx', 'vely', 'velz']
  quantities_to_extract += ['bcc1', 'bcc2', 'bcc3']
  if include_rad:
    quantities_to_extract += ['r00', 'r01', 'r02', 'r03', 'r11', 'r12', 'r13', 'r22', 'r23', 'r33']
    quantities_to_extract += ['r00_ff']

  # Parameters - quantities to save
  quantities_to_save = ['rho', 'ugas', 'ut', 'ur', 'uth', 'uph']
  quantities_to_save += ['umag', 'Br', 'Bth', 'Bph']
  if include_rad:
    quantities_to_save += \
        ['urad', 'Rtt', 'Rtr', 'Rtth', 'Rtph', 'Rrr', 'Rrth', 'Rrph', 'Rthth', 'Rthph', 'Rphph']

  # Calculate which frames to process
  frames = range(frame_min, frame_max + 1, frame_stride)
  if use_mpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    num_frames = len(frames)
    num_frames_per_rank = num_frames // size
    num_frames_extra = num_frames % size
    num_frames_list = [num_frames_per_rank + 1] * num_frames_extra \
        + [num_frames_per_rank] * (size - num_frames_extra)
    num_frames_previous = sum(num_frames_list[:rank])
    num_frames_local = num_frames_list[rank]
    frames = frames[num_frames_previous:num_frames_previous+num_frames_local]

  # Go through files
  for frame_n, frame in enumerate(frames):

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
          raise RuntimeError('Only 4- and 8-byte floating-point types supported for location data.')
        location_format = 'f' if location_size == 4 else 'd'
        if variable_size not in (4, 8):
          raise RuntimeError('Only 4- and 8-byte floating-point types supported for cell data.')
        variable_format = 'f' if variable_size == 4 else 'd'
        num_variables_base = len(variable_names_base)
        variable_names = []
        variable_inds = []
        for quantity in quantities_to_extract:
          if quantity not in variable_names_base:
            raise RuntimeError('Required variable {0} not found.'.format(quantity))
          variable_names.append(quantity)
          variable_ind = 0
          while variable_names_base[variable_ind] != quantity:
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

      # Extract adiabatic index from input file metadata
      if frame_n == 0:
        try:
          gamma_adi = float(input_data['mhd']['gamma'])
        except:
          raise RuntimeError('Unable to find adiabatic index in input file.')

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
            data_cks[name].append(np.array(struct.unpack(block_cell_format, \
                f.read(variable_data_size))).reshape(n_z, n_y, n_x))
          else:
            data_cks[name][block_n] = np.array(struct.unpack(block_cell_format, \
                f.read(variable_data_size))).reshape(n_z, n_y, n_x)
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
        cells_per_decade = cells_per_decade_default
        n_r = int(round(np.log10(r_max / r_min) * cells_per_decade))
      else:
        cells_per_decade = n_r / np.log10(r_max / r_min)
      if lat_max is None:
        th_min = 0.0
        th_max = np.pi
      else:
        th_min = np.pi/2.0 - lat_max * np.pi/180.0
        th_max = np.pi/2.0 + lat_max * np.pi/180.0
      if n_th is None:
        n_th = int(round((1.0 + 2.0 / r_square) ** -0.5 * cells_per_decade * np.log10(np.e) \
            * (th_max - th_min)))
      ph_min = 0.0
      ph_max = 2.0*np.pi
      if n_ph is None:
        n_ph = int(round(((1.0 + (1.0 + 2.0 / r_square) * a2 / r_square ** 2) \
            / (1.0 + 2.0 / r_square)) ** 0.5 * cells_per_decade * np.log10(np.e) * 2.0*np.pi))

    # Construct new grid
    if frame_n == 0:
      lrf = np.linspace(np.log(r_min), np.log(r_max), n_r + 1)
      lr = 0.5 * (lrf[:-1] + lrf[1:])
      rf = np.exp(lrf)
      r = np.exp(lr)[None,None,:]
      thf = np.linspace(th_min, th_max, n_th + 1)
      th = 0.5 * (thf[:-1] + thf[1:])[None,:,None]
      sth = np.sin(th)
      cth = np.cos(th)
      phf = np.linspace(ph_min, ph_max, n_ph + 1)
      ph = 0.5 * (phf[:-1] + phf[1:])[:,None,None]
      sph = np.sin(ph)
      cph = np.cos(ph)

    # Construct mapping
    if frame_n == 0:
      inds = np.empty((4, n_ph, n_th, n_r), dtype=int)
      if interp:
        weights = np.empty((3, n_ph, n_th, n_r))
      for ind_ph in range(n_ph):
        for ind_th in range(n_th):
          for ind_r in range(n_r):
            x_val = sth[0,ind_th,0] * (r[0,0,ind_r] * cph[ind_ph,0,0] - a * sph[ind_ph,0,0])
            y_val = sth[0,ind_th,0] * (r[0,0,ind_r] * sph[ind_ph,0,0] + a * cph[ind_ph,0,0])
            z_val = r[0,0,ind_r] * cth[0,ind_th,0]
            x_inds = (x_val >= block_lims[:,0]) & (x_val < block_lims[:,1])
            y_inds = (y_val >= block_lims[:,2]) & (y_val < block_lims[:,3])
            z_inds = (z_val >= block_lims[:,4]) & (z_val < block_lims[:,5])
            ind_b = np.where(x_inds & y_inds & z_inds)[0][0]
            inds[0,ind_ph,ind_th,ind_r] = ind_b
            if interp:
              ind_frac_x = (x_val - block_lims[ind_b,0]) \
                  / (block_lims[ind_b,1] - block_lims[ind_b,0]) * n_x - 0.5
              ind_frac_y = (y_val - block_lims[ind_b,2]) \
                  / (block_lims[ind_b,3] - block_lims[ind_b,2]) * n_y - 0.5
              ind_frac_z = (z_val - block_lims[ind_b,4]) \
                  / (block_lims[ind_b,5] - block_lims[ind_b,4]) * n_z - 0.5
              ind_x = min(int(ind_frac_x), n_x - 2)
              ind_y = min(int(ind_frac_y), n_y - 2)
              ind_z = min(int(ind_frac_z), n_z - 2)
              inds[1,ind_ph,ind_th,ind_r] = ind_z
              inds[2,ind_ph,ind_th,ind_r] = ind_y
              inds[3,ind_ph,ind_th,ind_r] = ind_x
              weights[0,ind_ph,ind_th,ind_r] = ind_frac_z - ind_z
              weights[1,ind_ph,ind_th,ind_r] = ind_frac_y - ind_y
              weights[2,ind_ph,ind_th,ind_r] = ind_frac_x - ind_x
            else:
              ind_frac_x = \
                  (x_val - block_lims[ind_b,0]) / (block_lims[ind_b,1] - block_lims[ind_b,0]) * n_x
              ind_frac_y = \
                  (y_val - block_lims[ind_b,2]) / (block_lims[ind_b,3] - block_lims[ind_b,2]) * n_y
              ind_frac_z = \
                  (z_val - block_lims[ind_b,4]) / (block_lims[ind_b,5] - block_lims[ind_b,4]) * n_z
              ind_x = min(int(ind_frac_x), n_x - 1)
              ind_y = min(int(ind_frac_y), n_y - 1)
              ind_z = min(int(ind_frac_z), n_z - 1)
              inds[1,ind_ph,ind_th,ind_r] = ind_z
              inds[2,ind_ph,ind_th,ind_r] = ind_y
              inds[3,ind_ph,ind_th,ind_r] = ind_x

    # Remap data to SKS grid
    if frame_n == 0:
      data_sks = {}
      for quantity in quantities_to_extract:
        data_sks[quantity] = np.empty((n_ph, n_th, n_r))
    for ind_ph in range(n_ph):
      for ind_th in range(n_th):
        for ind_r in range(n_r):
          ind_b = inds[0,ind_ph,ind_th,ind_r]
          ind_z = inds[1,ind_ph,ind_th,ind_r]
          ind_y = inds[2,ind_ph,ind_th,ind_r]
          ind_x = inds[3,ind_ph,ind_th,ind_r]
          if interp:
            weight_z = weights[0,ind_ph,ind_th,ind_r]
            weight_y = weights[1,ind_ph,ind_th,ind_r]
            weight_x = weights[2,ind_ph,ind_th,ind_r]
            for quantity in quantities_to_extract:
              val_mmm = data_cks[quantity][ind_b,ind_z,ind_y,ind_x]
              val_mmp = data_cks[quantity][ind_b,ind_z,ind_y,ind_x+1]
              val_mpm = data_cks[quantity][ind_b,ind_z,ind_y+1,ind_x]
              val_mpp = data_cks[quantity][ind_b,ind_z,ind_y+1,ind_x+1]
              val_pmm = data_cks[quantity][ind_b,ind_z+1,ind_y,ind_x]
              val_pmp = data_cks[quantity][ind_b,ind_z+1,ind_y,ind_x+1]
              val_ppm = data_cks[quantity][ind_b,ind_z+1,ind_y+1,ind_x]
              val_ppp = data_cks[quantity][ind_b,ind_z+1,ind_y+1,ind_x+1]
              data_sks[quantity][ind_ph,ind_th,ind_r] = (1.0 - weight_z) * ((1.0 - weight_y) \
                  * ((1.0 - weight_x) * val_mmm + weight_x * val_mmp) + weight_y * ((1.0 \
                  - weight_x) * val_mpm + weight_x * val_mpp)) + weight_z * ((1.0 - weight_y) \
                  * ((1.0 - weight_x) * val_pmm + weight_x * val_pmp) + weight_y * ((1.0 \
                  - weight_x) * val_ppm + weight_x * val_ppp))
          else:
            for quantity in quantities_to_extract:
              data_sks[quantity][ind_ph,ind_th,ind_r] = data_cks[quantity][ind_b,ind_z,ind_y,ind_x]

    # Calculate CKS coordinates
    if frame_n == 0:
      x = sth * (r * cph - a * sph)
      y = sth * (r * sph + a * cph)
      z = r * cth
      r2 = r ** 2
      x2 = x ** 2
      y2 = y ** 2
      z2 = z ** 2

    # Calculate SKS quantities
    if frame_n == 0:
      sth2 = sth ** 2
      sigma = r2 + a2 * cth ** 2
      f = 2.0 * r / sigma

    # Calculate SKS covariant metric
    if frame_n == 0:
      g_tt = -(1.0 - f)
      g_tr = f
      g_tph = -a * f * sth2
      g_rr = 1.0 + f
      g_rph = -(1.0 + f) * a * sth2
      g_thth = sigma
      g_phph = (r2 + a2 + a2 * f * sth2) * sth2

    # Calculate SKS contravariant metric
    if frame_n == 0:
      gtt = -(1.0 + f)

    # Calculate CKS quantities
    if frame_n == 0:
      lx = (r * x + a * y) / (r2 + a2)
      ly = (r * y - a * x) / (r2 + a2)
      lz = z / r

    # Calculate CKS covariant metric
    if frame_n == 0:
      g_xx = f * lx * lx + 1.0
      g_xy = f * lx * ly
      g_xz = f * lx * lz
      g_yy = f * ly * ly + 1.0
      g_yz = f * ly * lz
      g_zz = f * lz * lz + 1.0

    # Calculate CKS contravariant metric
    if frame_n == 0:
      gtx = f * lx
      gty = f * ly
      gtz = f * lz

    # Calculate CKS lapse and shift
    if frame_n == 0:
      alpha_coord = 1.0 / np.sqrt(-gtt)
      betax = -gtx / gtt
      betay = -gty / gtt
      betaz = -gtz / gtt

    # Calculate Jacobian
    if frame_n == 0:
      rr2 = x2 + y2 + z2
      rr = np.sqrt(rr2)
      drr_dx = x / rr
      drr_dy = y / rr
      drr_dz = z / rr
      dr_dx = rr * r * drr_dx / (2.0 * r2 - rr2 + a2)
      dr_dy = rr * r * drr_dy / (2.0 * r2 - rr2 + a2)
      dr_dz = (rr * r * drr_dz + a2 * z / r) / (2.0 * r2 - rr2 + a2)
      dth_dx = z / r * dr_dx / np.sqrt(r2 - z2)
      dth_dy = z / r * dr_dy / np.sqrt(r2 - z2)
      dth_dz = (z / r * dr_dz - 1.0) / np.sqrt(r2 - z2)
      dph_dx = -y / (x2 + y2) + a * dr_dx / (r2 + a2)
      dph_dy = x / (x2 + y2) + a * dr_dy / (r2 + a2)
      dph_dz = a * dr_dz / (r2 + a2)

    # Rename variables
    data_sks['rho'] = data_sks['dens']
    data_sks['ugas'] = data_sks['eint']
    data_sks['uux'] = data_sks['velx']
    data_sks['uuy'] = data_sks['vely']
    data_sks['uuz'] = data_sks['velz']
    data_sks['Bx'] = data_sks['bcc1']
    data_sks['By'] = data_sks['bcc2']
    data_sks['Bz'] = data_sks['bcc3']
    if include_rad:
      data_sks['urad'] = data_sks['r00_ff']
      data_sks['Rtt'] = data_sks['r00']
      data_sks['Rtx'] = data_sks['r01']
      data_sks['Rty'] = data_sks['r02']
      data_sks['Rtz'] = data_sks['r03']
      data_sks['Rxx'] = data_sks['r11']
      data_sks['Rxy'] = data_sks['r12']
      data_sks['Rxz'] = data_sks['r13']
      data_sks['Ryy'] = data_sks['r22']
      data_sks['Ryz'] = data_sks['r23']
      data_sks['Rzz'] = data_sks['r33']

    # Calculate velocities in CKS components
    uut = np.sqrt(1.0 + g_xx * data_sks['uux'] ** 2 \
        + 2.0 * g_xy * data_sks['uux'] * data_sks['uuy'] \
        + 2.0 * g_xz * data_sks['uux'] * data_sks['uuz'] + g_yy * data_sks['uuy'] ** 2 \
        + 2.0 * g_yz * data_sks['uuy'] * data_sks['uuz'] + g_zz * data_sks['uuz'] ** 2)
    data_sks['ut'] = uut / alpha_coord
    ux = data_sks['uux'] - betax * data_sks['ut']
    uy = data_sks['uuy'] - betay * data_sks['ut']
    uz = data_sks['uuz'] - betaz * data_sks['ut']

    # Calculate velocities in SKS components
    data_sks['ur'] = dr_dx * ux + dr_dy * uy + dr_dz * uz
    data_sks['uth'] = dth_dx * ux + dth_dy * uy + dth_dz * uz
    data_sks['uph'] = dph_dx * ux + dph_dy * uy + dph_dz * uz
    u_r = g_tr * data_sks['ut'] + g_rr * data_sks['ur'] + g_rph * data_sks['uph']
    u_th = g_thth * data_sks['uth']
    u_ph = g_tph * data_sks['ut'] + g_rph * data_sks['ur'] + g_phph * data_sks['uph']

    # Calculate magnetic fields in SKS components
    data_sks['Br'] = dr_dx * data_sks['Bx'] + dr_dy * data_sks['By'] + dr_dz * data_sks['Bz']
    data_sks['Bth'] = dth_dx * data_sks['Bx'] + dth_dy * data_sks['By'] + dth_dz * data_sks['Bz']
    data_sks['Bph'] = dph_dx * data_sks['Bx'] + dph_dy * data_sks['By'] + dph_dz * data_sks['Bz']
    bt = u_r * data_sks['Br'] + u_th * data_sks['Bth'] + u_ph * data_sks['Bph']
    br = (data_sks['Br'] + bt * data_sks['ur']) / data_sks['ut']
    bth = (data_sks['Bth'] + bt * data_sks['uth']) / data_sks['ut']
    bph = (data_sks['Bph'] + bt * data_sks['uph']) / data_sks['ut']
    b_t = g_tt * bt + g_tr * br + g_tph * bph
    b_r = g_tr * bt + g_rr * br + g_rph * bph
    b_th = g_thth * bth
    b_ph = g_tph * bt + g_rph * br + g_phph * bph
    data_sks['umag'] = 0.5 * (b_t * bt + b_r * br + b_th * bth + b_ph * bph)

    # Calculate radiation stress-energy tensor in SKS components
    if include_rad:
      data_sks['Rtr'] = dr_dx * data_sks['Rtx'] + dr_dy * data_sks['Rty'] + dr_dz * data_sks['Rtz']
      data_sks['Rtth'] = \
          dth_dx * data_sks['Rtx'] + dth_dy * data_sks['Rty'] + dth_dz * data_sks['Rtz']
      data_sks['Rtph'] = \
          dph_dx * data_sks['Rtx'] + dph_dy * data_sks['Rty'] + dph_dz * data_sks['Rtz']
      data_sks['Rrr'] = dr_dx * dr_dx * data_sks['Rxx'] \
          + (dr_dx * dr_dy + dr_dy * dr_dx) * data_sks['Rxy'] \
          + (dr_dx * dr_dz + dr_dz * dr_dx) * data_sks['Rxz'] + dr_dy * dr_dy * data_sks['Ryy'] \
          + (dr_dy * dr_dz + dr_dz * dr_dy) * data_sks['Ryz'] + dr_dz * dr_dz * data_sks['Rzz']
      data_sks['Rrth'] = dr_dx * dth_dx * data_sks['Rxx'] \
          + (dr_dx * dth_dy + dr_dy * dth_dx) * data_sks['Rxy'] \
          + (dr_dx * dth_dz + dr_dz * dth_dx) * data_sks['Rxz'] + dr_dy * dth_dy * data_sks['Ryy'] \
          + (dr_dy * dth_dz + dr_dz * dth_dy) * data_sks['Ryz'] + dr_dz * dth_dz * data_sks['Rzz']
      data_sks['Rrph'] = dr_dx * dph_dx * data_sks['Rxx'] \
          + (dr_dx * dph_dy + dr_dy * dph_dx) * data_sks['Rxy'] \
          + (dr_dx * dph_dz + dr_dz * dph_dx) * data_sks['Rxz'] + dr_dy * dph_dy * data_sks['Ryy'] \
          + (dr_dy * dph_dz + dr_dz * dph_dy) * data_sks['Ryz'] + dr_dz * dph_dz * data_sks['Rzz']
      data_sks['Rthth'] = dth_dx * dth_dx * data_sks['Rxx'] \
          + (dth_dx * dth_dy + dth_dy * dth_dx) * data_sks['Rxy'] \
          + (dth_dx * dth_dz + dth_dz * dth_dx) * data_sks['Rxz'] \
          + dth_dy * dth_dy * data_sks['Ryy'] \
          + (dth_dy * dth_dz + dth_dz * dth_dy) * data_sks['Ryz'] \
          + dth_dz * dth_dz * data_sks['Rzz']
      data_sks['Rthph'] = dth_dx * dph_dx * data_sks['Rxx'] \
          + (dth_dx * dph_dy + dth_dy * dph_dx) * data_sks['Rxy'] \
          + (dth_dx * dph_dz + dth_dz * dph_dx) * data_sks['Rxz'] \
          + dth_dy * dph_dy * data_sks['Ryy'] \
          + (dth_dy * dph_dz + dth_dz * dph_dy) * data_sks['Ryz'] \
          + dth_dz * dph_dz * data_sks['Rzz']
      data_sks['Rphph'] = dph_dx * dph_dx * data_sks['Rxx'] \
          + (dph_dx * dph_dy + dph_dy * dph_dx) * data_sks['Rxy'] \
          + (dph_dx * dph_dz + dph_dz * dph_dx) * data_sks['Rxz'] \
          + dph_dy * dph_dy * data_sks['Ryy'] \
          + (dph_dy * dph_dz + dph_dz * dph_dy) * data_sks['Ryz'] \
          + dph_dz * dph_dz * data_sks['Rzz']

    # Save results
    data_out = {}
    data_out['a'] = a
    data_out['gamma_adi'] = gamma_adi
    data_out['rf'] = rf
    data_out['r'] = r[0,0,:]
    data_out['thf'] = thf
    data_out['th'] = th[0,:,0]
    data_out['phf'] = phf
    data_out['ph'] = ph[:,0,0]
    for quantity in quantities_to_save:
      data_out[quantity] = data_sks[quantity]
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
  parser.add_argument('--lat_max', type=float, \
      help='maximum latitude (degrees) grid extends away from midplane')
  parser.add_argument('--n_th', type=int, help='number of polar zones in output grid')
  parser.add_argument('--n_ph', type=int, \
      help='number of azimuthal zones in 3D version of output grid')
  parser.add_argument('--no_interp', action='store_true', \
      help='flag indicating remapping to be done with nearest neighbors rather than interpolation')
  parser.add_argument('--no_rad', action='store_true', \
      help='flag indicating radiation quantities should be ignored')
  parser.add_argument('--mpi', action='store_true', \
      help='flag indicating MPI should be used to distribute work')
  args = parser.parse_args()
  main(**vars(args))
