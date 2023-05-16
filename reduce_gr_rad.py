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
  --no_interp: flag indicating remapping to be done with nearest neighbors
      rather than interpolation

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
import warnings

# Numerical modules
import numpy as np

# Main function
def main(**kwargs):

  # Parameters - fixed
  cells_per_decade = 32.0
  r_square = 10.0

  # Parameters - quantities to extract
  quantities_to_extract = ['dens', 'eint', 'velx', 'vely', 'velz']
  quantities_to_extract += ['bcc1', 'bcc2', 'bcc3']
  quantities_to_extract += ['r00', 'r01', 'r02', 'r03', 'r11', 'r12', 'r13', 'r22', 'r23', 'r33']
  quantities_to_extract += ['r00_ff']

  # Parameters - quantities to average
  quantities_to_average = ['rho', 'pgas', 'T_cgs']
  quantities_to_average += ['pmag', 'beta_inv', 'sigma']
  quantities_to_average += ['prad', 'prad_rho', 'prad_pgas', 'pmag_prad']
  quantities_to_average += ['pgas_ptot', 'pmag_ptot', 'prad_ptot']
  quantities_to_average += ['uut', 'ut', 'ur', 'uth', 'uph', 'vr', 'vth', 'vph']
  quantities_to_average += ['Br', 'Bth', 'Bph']
  quantities_to_average += ['acc_r_tot']
  quantities_to_average += ['acc_r_pgas', 'acc_r_pmag', 'acc_r_prad']
  quantities_to_average += ['acc_r_tens', 'acc_r_visc']
  quantities_to_average += ['acc_r_grav', 'acc_r_cent', 'acc_r_gr']
  quantities_to_average += ['acc_r_pgas_other', 'acc_r_pmag_other', 'acc_r_prad_other']
  quantities_to_average += ['acc_r_mag_other', 'acc_r_rad_other']
  quantities_to_average += ['acc_th_tot']
  quantities_to_average += ['acc_th_pgas', 'acc_th_pmag', 'acc_th_prad']
  quantities_to_average += ['acc_th_tens', 'acc_th_visc']
  quantities_to_average += ['acc_th_cent', 'acc_th_gr']
  quantities_to_average += ['acc_th_pgas_other', 'acc_th_pmag_other', 'acc_th_prad_other']
  quantities_to_average += ['acc_th_mag_other', 'acc_th_rad_other']
  quantities_to_average += ['Tgas_rph_f', 'Tgas_thph_f']
  quantities_to_average += ['Tmag_rph_f', 'Tmag_thph_f']
  quantities_to_average += ['Tradtr', 'Tradtth', 'Trad_rph_f', 'Trad_thph_f']

  # Parameters - quantities to save
  quantities_to_save = list(quantities_to_average)
  quantities_to_save += ['ugas', 'umag', 'urad']

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

  # Parameters - physical units
  c_cgs = 2.99792458e10
  kb_cgs = 1.380649e-16
  mp_cgs = 1.67262192369e-24

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

      # Extract molecular weight from input file metadata
      if frame_n == 0:
        try:
          mu = float(input_data['units']['mu'])
        except:
          raise RuntimeError('Unable to find molecular weight in input file.')

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

    # Remap data to 3D SKS grid
    if frame_n == 0:
      data_3d = {}
      for quantity in quantities_to_extract:
        data_3d[quantity] = np.empty((n_ph, n_th, n_r))
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
              data_3d[quantity][ind_ph,ind_th,ind_r] = (1.0 - weight_z) * ((1.0 - weight_y) \
                  * ((1.0 - weight_x) * val_mmm + weight_x * val_mmp) + weight_y * ((1.0 \
                  - weight_x) * val_mpm + weight_x * val_mpp)) + weight_z * ((1.0 - weight_y) \
                  * ((1.0 - weight_x) * val_pmm + weight_x * val_pmp) + weight_y * ((1.0 \
                  - weight_x) * val_ppm + weight_x * val_ppp))
          else:
            for quantity in quantities_to_extract:
              data_3d[quantity][ind_ph,ind_th,ind_r] = data_cks[quantity][ind_b,ind_z,ind_y,ind_x]

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
      cth2 = cth ** 2
      delta = r2 - 2.0 * r + a2
      sigma = r2 + a2 * cth2
      sigma2 = sigma ** 2
      sigma_alt = r2 - a2 * cth2
      f = 2.0 * r / sigma
      det = sigma * sth

    # Calculate SKS covariant metric
    if frame_n == 0:
      g_tt = -(1.0 - f)
      g_tr = f
      g_tth = 0.0
      g_tph = -a * f * sth2
      g_rr = 1.0 + f
      g_rth = 0.0
      g_rph = -(1.0 + f) * a * sth2
      g_thth = sigma
      g_thph = 0.0
      g_phph = (r2 + a2 + a2 * f * sth2) * sth2

    # Calculate SKS covariant metric derivatives
    if frame_n == 0:
      dr_g_tt = -2.0 * sigma_alt / sigma2
      dr_g_tr = -2.0 * sigma_alt / sigma2
      dr_g_tth = 0.0
      dr_g_tph = 2.0 * sigma_alt / sigma2 * a * sth2
      dr_g_rr = -2.0 * sigma_alt / sigma2
      dr_g_rth = 0.0
      dr_g_rph = 2.0 * sigma_alt / sigma2 * a * sth2
      dr_g_thth = 2.0 * r
      dr_g_thph = 0.0
      dr_g_phph = 2.0 * (r - sigma_alt / sigma2 * a2 * sth2) * sth2
      dth_g_tt = 4.0 * a2 * r / sigma2 * sth * cth;
      dth_g_tr = 4.0 * a2 * r / sigma2 * sth * cth;
      dth_g_tth = 0.0
      dth_g_tph = -4.0 * a * r * (r2 + a2) / sigma2 * sth * cth;
      dth_g_rr = 4.0 * a2 * r / sigma2 * sth * cth;
      dth_g_rth = 0.0
      dth_g_rph = -2.0 * (1.0 + 2.0 * r * (r2 + a2) / sigma2) * a * sth * cth;
      dth_g_thth = -2.0 * a2 * sth * cth;
      dth_g_thph = 0.0
      dth_g_phph = 2.0 * (delta + 2.0 * r * (r2 + a2) ** 2 / sigma2) * sth * cth;

    # Calculate SKS contravariant metric
    if frame_n == 0:
      gtt = -(1.0 + f)
      gtr = f
      gtth = 0.0
      gtph = 0.0
      grr = delta / sigma
      grth = 0.0
      grph = a / sigma
      gthth = 1.0 / sigma
      gthph = 0.0
      gphph = 1.0 / (sigma * sth2)

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
    data_3d['rho'] = data_3d['dens']
    data_3d['ugas'] = data_3d['eint']
    data_3d['uux'] = data_3d['velx']
    data_3d['uuy'] = data_3d['vely']
    data_3d['uuz'] = data_3d['velz']
    data_3d['Bx'] = data_3d['bcc1']
    data_3d['By'] = data_3d['bcc2']
    data_3d['Bz'] = data_3d['bcc3']
    data_3d['urad'] = data_3d['r00_ff']
    data_3d['Tradtt'] = data_3d['r00']
    data_3d['Tradtx'] = data_3d['r01']
    data_3d['Tradty'] = data_3d['r02']
    data_3d['Tradtz'] = data_3d['r03']
    data_3d['Tradxx'] = data_3d['r11']
    data_3d['Tradxy'] = data_3d['r12']
    data_3d['Tradxz'] = data_3d['r13']
    data_3d['Tradyy'] = data_3d['r22']
    data_3d['Tradyz'] = data_3d['r23']
    data_3d['Tradzz'] = data_3d['r33']

    # Calculate velocities in CKS components
    data_3d['uut'] = np.sqrt(1.0 + g_xx * data_3d['uux'] ** 2 \
        + 2.0 * g_xy * data_3d['uux'] * data_3d['uuy'] \
        + 2.0 * g_xz * data_3d['uux'] * data_3d['uuz'] + g_yy * data_3d['uuy'] ** 2 \
        + 2.0 * g_yz * data_3d['uuy'] * data_3d['uuz'] + g_zz * data_3d['uuz'] ** 2)
    data_3d['ut'] = data_3d['uut'] / alpha_coord
    data_3d['ux'] = data_3d['uux'] - betax * data_3d['ut']
    data_3d['uy'] = data_3d['uuy'] - betay * data_3d['ut']
    data_3d['uz'] = data_3d['uuz'] - betaz * data_3d['ut']

    # Calculate velocities in SKS components
    data_3d['ur'] = dr_dx * data_3d['ux'] + dr_dy * data_3d['uy'] + dr_dz * data_3d['uz']
    data_3d['uth'] = dth_dx * data_3d['ux'] + dth_dy * data_3d['uy'] + dth_dz * data_3d['uz']
    data_3d['uph'] = dph_dx * data_3d['ux'] + dph_dy * data_3d['uy'] + dph_dz * data_3d['uz']
    u_t = g_tt * data_3d['ut'] + g_tr * data_3d['ur'] + g_tph * data_3d['uph']
    u_r = g_tr * data_3d['ut'] + g_rr * data_3d['ur'] + g_rph * data_3d['uph']
    u_th = g_thth * data_3d['uth']
    u_ph = g_tph * data_3d['ut'] + g_rph * data_3d['ur'] + g_phph * data_3d['uph']

    # Calculate magnetic fields in SKS components
    data_3d['Br'] = dr_dx * data_3d['Bx'] + dr_dy * data_3d['By'] + dr_dz * data_3d['Bz']
    data_3d['Bth'] = dth_dx * data_3d['Bx'] + dth_dy * data_3d['By'] + dth_dz * data_3d['Bz']
    data_3d['Bph'] = dph_dx * data_3d['Bx'] + dph_dy * data_3d['By'] + dph_dz * data_3d['Bz']
    bt = u_r * data_3d['Br'] + u_th * data_3d['Bth'] + u_ph * data_3d['Bph']
    br = (data_3d['Br'] + bt * data_3d['ur']) / data_3d['ut']
    bth = (data_3d['Bth'] + bt * data_3d['uth']) / data_3d['ut']
    bph = (data_3d['Bph'] + bt * data_3d['uph']) / data_3d['ut']
    b_t = g_tt * bt + g_tr * br + g_tph * bph
    b_r = g_tr * bt + g_rr * br + g_rph * bph
    b_th = g_thth * bth
    b_ph = g_tph * bt + g_rph * br + g_phph * bph

    # Calculate contravariant radiation stress-energy tensor in SKS components
    data_3d['Tradtr'] = \
        dr_dx * data_3d['Tradtx'] + dr_dy * data_3d['Tradty'] + dr_dz * data_3d['Tradtz']
    data_3d['Tradtth'] = \
        dth_dx * data_3d['Tradtx'] + dth_dy * data_3d['Tradty'] + dth_dz * data_3d['Tradtz']
    data_3d['Tradtph'] = \
        dph_dx * data_3d['Tradtx'] + dph_dy * data_3d['Tradty'] + dph_dz * data_3d['Tradtz']
    data_3d['Tradrr'] = dr_dx * dr_dx * data_3d['Tradxx'] \
        + (dr_dx * dr_dy + dr_dy * dr_dx) * data_3d['Tradxy'] \
        + (dr_dx * dr_dz + dr_dz * dr_dx) * data_3d['Tradxz'] + dr_dy * dr_dy * data_3d['Tradyy'] \
        + (dr_dy * dr_dz + dr_dz * dr_dy) * data_3d['Tradyz'] + dr_dz * dr_dz * data_3d['Tradzz']
    data_3d['Tradrth'] = dr_dx * dth_dx * data_3d['Tradxx'] \
        + (dr_dx * dth_dy + dr_dy * dth_dx) * data_3d['Tradxy'] \
        + (dr_dx * dth_dz + dr_dz * dth_dx) * data_3d['Tradxz'] \
        + dr_dy * dth_dy * data_3d['Tradyy'] \
        + (dr_dy * dth_dz + dr_dz * dth_dy) * data_3d['Tradyz'] + dr_dz * dth_dz * data_3d['Tradzz']
    data_3d['Tradrph'] = dr_dx * dph_dx * data_3d['Tradxx'] \
        + (dr_dx * dph_dy + dr_dy * dph_dx) * data_3d['Tradxy'] \
        + (dr_dx * dph_dz + dr_dz * dph_dx) * data_3d['Tradxz'] \
        + dr_dy * dph_dy * data_3d['Tradyy'] \
        + (dr_dy * dph_dz + dr_dz * dph_dy) * data_3d['Tradyz'] + dr_dz * dph_dz * data_3d['Tradzz']
    data_3d['Tradthth'] = dth_dx * dth_dx * data_3d['Tradxx'] \
        + (dth_dx * dth_dy + dth_dy * dth_dx) * data_3d['Tradxy'] \
        + (dth_dx * dth_dz + dth_dz * dth_dx) * data_3d['Tradxz'] \
        + dth_dy * dth_dy * data_3d['Tradyy'] \
        + (dth_dy * dth_dz + dth_dz * dth_dy) * data_3d['Tradyz'] \
        + dth_dz * dth_dz * data_3d['Tradzz']
    data_3d['Tradthph'] = dth_dx * dph_dx * data_3d['Tradxx'] \
        + (dth_dx * dph_dy + dth_dy * dph_dx) * data_3d['Tradxy'] \
        + (dth_dx * dph_dz + dth_dz * dph_dx) * data_3d['Tradxz'] \
        + dth_dy * dph_dy * data_3d['Tradyy'] \
        + (dth_dy * dph_dz + dth_dz * dph_dy) * data_3d['Tradyz'] \
        + dth_dz * dph_dz * data_3d['Tradzz']
    data_3d['Tradphph'] = dph_dx * dph_dx * data_3d['Tradxx'] \
        + (dph_dx * dph_dy + dph_dy * dph_dx) * data_3d['Tradxy'] \
        + (dph_dx * dph_dz + dph_dz * dph_dx) * data_3d['Tradxz'] \
        + dph_dy * dph_dy * data_3d['Tradyy'] \
        + (dph_dy * dph_dz + dph_dz * dph_dy) * data_3d['Tradyz'] \
        + dph_dz * dph_dz * data_3d['Tradzz']

    # Calculate covariant radiation stress-energy tensor in SKS components
    ttrad_tt = g_tt * g_tt * data_3d['Tradtt'] + (g_tt * g_tr + g_tr * g_tt) * data_3d['Tradtr'] \
        + (g_tt * g_tth + g_tth * g_tt) * data_3d['Tradtth'] \
        + (g_tt * g_tph + g_tph * g_tt) * data_3d['Tradtph'] + g_tr * g_tr * data_3d['Tradrr'] \
        + (g_tr * g_tth + g_tth * g_tr) * data_3d['Tradrth'] \
        + (g_tr * g_tph + g_tph * g_tr) * data_3d['Tradrph'] + g_tth * g_tth * data_3d['Tradthth'] \
        + (g_tth * g_tph + g_tph * g_tth) * data_3d['Tradthph'] \
        + g_tph * g_tph * data_3d['Tradphph']
    ttrad_tr = g_tt * g_tr * data_3d['Tradtt'] + (g_tt * g_rr + g_tr * g_tr) * data_3d['Tradtr'] \
        + (g_tt * g_rth + g_tth * g_tr) * data_3d['Tradtth'] \
        + (g_tt * g_rph + g_tph * g_tr) * data_3d['Tradtph'] + g_tr * g_rr * data_3d['Tradrr'] \
        + (g_tr * g_rth + g_tth * g_rr) * data_3d['Tradrth'] \
        + (g_tr * g_rph + g_tph * g_rr) * data_3d['Tradrph'] + g_tth * g_rth * data_3d['Tradthth'] \
        + (g_tth * g_rph + g_tph * g_rth) * data_3d['Tradthph'] \
        + g_tph * g_rph * data_3d['Tradphph']
    ttrad_tth = g_tt * g_tth * data_3d['Tradtt'] \
        + (g_tt * g_rth + g_tr * g_tth) * data_3d['Tradtr'] \
        + (g_tt * g_thth + g_tth * g_tth) * data_3d['Tradtth'] \
        + (g_tt * g_thph + g_tph * g_tth) * data_3d['Tradtph'] + g_tr * g_rth * data_3d['Tradrr'] \
        + (g_tr * g_thth + g_tth * g_rth) * data_3d['Tradrth'] \
        + (g_tr * g_thph + g_tph * g_rth) * data_3d['Tradrph'] \
        + g_tth * g_thth * data_3d['Tradthth'] \
        + (g_tth * g_thph + g_tph * g_thth) * data_3d['Tradthph'] \
        + g_tph * g_thph * data_3d['Tradphph']
    ttrad_tph = g_tt * g_tph * data_3d['Tradtt'] \
        + (g_tt * g_rph + g_tr * g_tph) * data_3d['Tradtr'] \
        + (g_tt * g_thph + g_tth * g_tph) * data_3d['Tradtth'] \
        + (g_tt * g_phph + g_tph * g_tph) * data_3d['Tradtph'] + g_tr * g_rph * data_3d['Tradrr'] \
        + (g_tr * g_thph + g_tth * g_rph) * data_3d['Tradrth'] \
        + (g_tr * g_phph + g_tph * g_rph) * data_3d['Tradrph'] \
        + g_tth * g_thph * data_3d['Tradthth'] \
        + (g_tth * g_phph + g_tph * g_thph) * data_3d['Tradthph'] \
        + g_tph * g_phph * data_3d['Tradphph']
    ttrad_rr = g_tr * g_tr * data_3d['Tradtt'] + (g_tr * g_rr + g_rr * g_tr) * data_3d['Tradtr'] \
        + (g_tr * g_rth + g_rth * g_tr) * data_3d['Tradtth'] \
        + (g_tr * g_rph + g_rph * g_tr) * data_3d['Tradtph'] + g_rr * g_rr * data_3d['Tradrr'] \
        + (g_rr * g_rth + g_rth * g_rr) * data_3d['Tradrth'] \
        + (g_rr * g_rph + g_rph * g_rr) * data_3d['Tradrph'] + g_rth * g_rth * data_3d['Tradthth'] \
        + (g_rth * g_rph + g_rph * g_rth) * data_3d['Tradthph'] \
        + g_rph * g_rph * data_3d['Tradphph']
    ttrad_rth = g_tr * g_tth * data_3d['Tradtt'] \
        + (g_tr * g_rth + g_rr * g_tth) * data_3d['Tradtr'] \
        + (g_tr * g_thth + g_rth * g_tth) * data_3d['Tradtth'] \
        + (g_tr * g_thph + g_rph * g_tth) * data_3d['Tradtph'] + g_rr * g_rth * data_3d['Tradrr'] \
        + (g_rr * g_thth + g_rth * g_rth) * data_3d['Tradrth'] \
        + (g_rr * g_thph + g_rph * g_rth) * data_3d['Tradrph'] \
        + g_rth * g_thth * data_3d['Tradthth'] \
        + (g_rth * g_thph + g_rph * g_thth) * data_3d['Tradthph'] \
        + g_rph * g_thph * data_3d['Tradphph']
    ttrad_rph = g_tr * g_tph * data_3d['Tradtt'] \
        + (g_tr * g_rph + g_rr * g_tph) * data_3d['Tradtr'] \
        + (g_tr * g_thph + g_rth * g_tph) * data_3d['Tradtth'] \
        + (g_tr * g_phph + g_rph * g_tph) * data_3d['Tradtph'] + g_rr * g_rph * data_3d['Tradrr'] \
        + (g_rr * g_thph + g_rth * g_rph) * data_3d['Tradrth'] \
        + (g_rr * g_phph + g_rph * g_rph) * data_3d['Tradrph'] \
        + g_rth * g_thph * data_3d['Tradthth'] \
        + (g_rth * g_phph + g_rph * g_thph) * data_3d['Tradthph'] \
        + g_rph * g_phph * data_3d['Tradphph']
    ttrad_thth = g_tth * g_tth * data_3d['Tradtt'] \
        + (g_tth * g_rth + g_rth * g_tth) * data_3d['Tradtr'] \
        + (g_tth * g_thth + g_thth * g_tth) * data_3d['Tradtth'] \
        + (g_tth * g_thph + g_thph * g_tth) * data_3d['Tradtph'] \
        + g_rth * g_rth * data_3d['Tradrr'] \
        + (g_rth * g_thth + g_thth * g_rth) * data_3d['Tradrth'] \
        + (g_rth * g_thph + g_thph * g_rth) * data_3d['Tradrph'] \
        + g_thth * g_thth * data_3d['Tradthth'] \
        + (g_thth * g_thph + g_thph * g_thth) * data_3d['Tradthph'] \
        + g_thph * g_thph * data_3d['Tradphph']
    ttrad_thph = g_tth * g_tph * data_3d['Tradtt'] \
        + (g_tth * g_rph + g_rth * g_tph) * data_3d['Tradtr'] \
        + (g_tth * g_thph + g_thth * g_tph) * data_3d['Tradtth'] \
        + (g_tth * g_phph + g_thph * g_tph) * data_3d['Tradtph'] \
        + g_rth * g_rph * data_3d['Tradrr'] \
        + (g_rth * g_thph + g_thth * g_rph) * data_3d['Tradrth'] \
        + (g_rth * g_phph + g_thph * g_rph) * data_3d['Tradrph'] \
        + g_thth * g_thph * data_3d['Tradthth'] \
        + (g_thth * g_phph + g_thph * g_thph) * data_3d['Tradthph'] \
        + g_thph * g_phph * data_3d['Tradphph']
    ttrad_phph = g_tph * g_tph * data_3d['Tradtt'] \
        + (g_tph * g_rph + g_rph * g_tph) * data_3d['Tradtr'] \
        + (g_tph * g_thph + g_thph * g_tph) * data_3d['Tradtth'] \
        + (g_tph * g_phph + g_phph * g_tph) * data_3d['Tradtph'] \
        + g_rph * g_rph * data_3d['Tradrr'] \
        + (g_rph * g_thph + g_thph * g_rph) * data_3d['Tradrth'] \
        + (g_rph * g_phph + g_phph * g_rph) * data_3d['Tradrph'] \
        + g_thph * g_thph * data_3d['Tradthth'] \
        + (g_thph * g_phph + g_phph * g_thph) * data_3d['Tradthph'] \
        + g_phph * g_phph * data_3d['Tradphph']

    # Calculate pre-averaging derived quantities
    with warnings.catch_warnings():

      # Ignore warnings
      warnings.filterwarnings('ignore', 'invalid value encountered in true_divide', RuntimeWarning)
      warnings.filterwarnings('ignore', 'invalid value encountered in sqrt', RuntimeWarning)

      # Calculate gas quantities
      data_3d['pgas'] = (gamma_adi - 1.0) * data_3d['ugas']
      data_3d['T_cgs'] = mu * mp_cgs * c_cgs ** 2 / kb_cgs * data_3d['pgas'] / data_3d['rho']
      data_3d['vr'] = data_3d['ur'] / data_3d['ut']
      data_3d['vth'] = data_3d['uth'] / data_3d['ut']
      data_3d['vph'] = data_3d['uph'] / data_3d['ut']

      # Calculate magnetic quantities
      data_3d['pmag'] = 0.5 * (b_t * bt + b_r * br + b_th * bth + b_ph * bph)
      data_3d['beta_inv'] = data_3d['pmag'] / data_3d['pgas']
      data_3d['sigma'] = 2.0 * data_3d['pmag'] / data_3d['rho']

      # Calculate radiation quantities
      data_3d['prad'] = 1.0/3.0 * data_3d['urad']
      data_3d['prad_rho'] = data_3d['prad'] / data_3d['rho']

      # Calculate pressures
      data_3d['prad_pgas'] = data_3d['prad'] / data_3d['pgas']
      data_3d['pmag_prad'] = data_3d['pmag'] / data_3d['prad']
      ptot = data_3d['pgas'] + data_3d['pmag'] + data_3d['prad']
      data_3d['pgas_ptot'] = data_3d['pgas'] / ptot
      data_3d['pmag_ptot'] = data_3d['pmag'] / ptot
      data_3d['prad_ptot'] = data_3d['prad'] / ptot

      # Calculate enthalpies
      wgas = data_3d['rho'] + data_3d['ugas'] + data_3d['pgas']
      wtot = wgas + 2.0 * data_3d['pmag'] + 4.0 * data_3d['prad']

      # Calculate radiation fluxes
      fradt = (data_3d['Tradtt'] - (data_3d['urad'] + data_3d['prad']) * data_3d['ut'] ** 2 \
          - data_3d['prad'] * gtt) / (2.0 * data_3d['ut'])
      fradr = (data_3d['Tradtr'] \
          - (data_3d['urad'] + data_3d['prad']) * data_3d['ut'] * data_3d['ur'] \
          - data_3d['prad'] * gtr - fradt * data_3d['ur']) / data_3d['ut']
      fradth = (data_3d['Tradtth'] - (data_3d['urad'] \
          + data_3d['prad']) * data_3d['ut'] * data_3d['uth'] - data_3d['prad'] * gtth \
          - fradt * data_3d['uth']) / data_3d['ut']
      fradph = (data_3d['Tradtph'] - (data_3d['urad'] \
          + data_3d['prad']) * data_3d['ut'] * data_3d['uph'] - data_3d['prad'] * gtph \
          - fradt * data_3d['uph']) / data_3d['ut']
      frad_r = g_tr * fradt + g_rr * fradr + g_rth * fradth + g_rph * fradph
      frad_th = g_tth * fradt + g_rth * fradr + g_thth * fradth + g_thph * fradph

      # Calculate radial accelerations
      data_3d['acc_r_tot'] = (np.gradient(det * wtot * data_3d['ur'] * u_r, r[0,0,:], axis=2) \
          + np.gradient(det * wtot * data_3d['uth'] * u_r, th[0,:,0], axis=1)) / (det * wtot)
      data_3d['acc_r_pgas'] = -np.gradient(det * data_3d['pgas'], r[0,0,:], axis=2) / (det * wtot)
      data_3d['acc_r_pmag'] = -np.gradient(det * data_3d['pmag'], r[0,0,:], axis=2) / (det * wtot)
      data_3d['acc_r_prad'] = -np.gradient(det * data_3d['prad'], r[0,0,:], axis=2) / (det * wtot)
      data_3d['acc_r_tens'] = (np.gradient(det * br * b_r, r[0,0,:], axis=2) \
          + np.gradient(det * bth * b_r, th[0,:,0], axis=1)) / (det * wtot)
      data_3d['acc_r_visc'] = \
          -(np.gradient(det * (fradr * u_r + data_3d['ur'] * frad_r), r[0,0,:], axis=2) \
          + np.gradient(det * (fradth * u_r + data_3d['uth'] * frad_r), th[0,:,0], axis=1)) \
          / (det * wtot)
      data_3d['acc_r_grav'] = 0.5 * dr_g_tt * data_3d['ut'] ** 2
      data_3d['acc_r_cent'] = \
          0.5 * (dr_g_thth * data_3d['uth'] ** 2 + dr_g_phph * data_3d['uph'] ** 2)
      data_3d['acc_r_gr'] = 0.5 * (dr_g_rr * data_3d['ur'] ** 2 \
          + 2.0 * dr_g_tr * data_3d['ut'] * data_3d['ur'] \
          + 2.0 * dr_g_tth * data_3d['ut'] * data_3d['uth'] \
          + 2.0 * dr_g_tph * data_3d['ut'] * data_3d['uph'] \
          + 2.0 * dr_g_rth * data_3d['ur'] * data_3d['uth'] \
          + 2.0 * dr_g_rph * data_3d['ur'] * data_3d['uph'] \
          + 2.0 * dr_g_thph * data_3d['uth'] * data_3d['uph'])
      temp = dr_g_tt * gtt + 2.0 * dr_g_tr * gtr + 2.0 * dr_g_tth * gtth + 2.0 * dr_g_tph * gtph \
          + dr_g_rr * grr + 2.0 * dr_g_rth * grth + 2.0 * dr_g_rph * grph + dr_g_thth * gthth \
          + 2.0 * dr_g_thph * gthph + dr_g_phph * gphph
      data_3d['acc_r_pgas_other'] = temp * data_3d['pgas'] / (2.0 * wtot)
      data_3d['acc_r_pmag_other'] = temp * data_3d['pmag'] / (2.0 * wtot)
      data_3d['acc_r_prad_other'] = temp * data_3d['prad'] / (2.0 * wtot)
      data_3d['acc_r_mag_other'] = -(dr_g_tt * bt ** 2 + 2.0 * dr_g_tr * bt * br \
          + 2.0 * dr_g_tth * bt * bth + 2.0 * dr_g_tph * bt * bph + dr_g_rr * br ** 2 \
          + 2.0 * dr_g_rth * br * bth + 2.0 * dr_g_rph * br * bph + dr_g_thth * bth ** 2 \
          + 2.0 * dr_g_thph * bth * bph + dr_g_phph * bph ** 2) / (2.0 * wtot)
      data_3d['acc_r_rad_other'] = dr_g_tt * fradt * data_3d['ut'] \
          + dr_g_tr * fradt * data_3d['ur'] + dr_g_tth * fradt * data_3d['uth'] \
          + dr_g_tph * fradt * data_3d['uph'] + dr_g_rr * fradr * data_3d['ur'] \
          + dr_g_rth * fradr * data_3d['uth'] + dr_g_rph * fradr * data_3d['uph'] \
          + dr_g_thth * fradth * data_3d['uth'] + dr_g_thph * fradth * data_3d['uph'] \
          + dr_g_phph * fradph * data_3d['uph']
      data_3d['acc_r_rad_other'] += dr_g_tt * data_3d['ut'] * fradt \
          + 2.0 * dr_g_tr * data_3d['ut'] * fradr + 2.0 * dr_g_tth * data_3d['ut'] * fradth \
          + 2.0 * dr_g_tph * data_3d['ut'] * fradph + dr_g_rr * data_3d['ur'] * fradr \
          + 2.0 * dr_g_rth * data_3d['ur'] * fradth + 2.0 * dr_g_rph * data_3d['ur'] * fradph \
          + dr_g_thth * data_3d['uth'] * fradth + 2.0 * dr_g_thph * data_3d['uth'] * fradph \
          + dr_g_phph * data_3d['uph'] * fradph
      data_3d['acc_r_rad_other'] /= 2.0 * wtot

      # Calculate polar accelerations
      data_3d['acc_th_tot'] = (np.gradient(det * wtot * data_3d['ur'] * u_th, r[0,0,:], axis=2) \
          + np.gradient(det * wtot * data_3d['uth'] * u_th, th[0,:,0], axis=1)) / (det * wtot)
      data_3d['acc_th_pgas'] = -np.gradient(det * data_3d['pgas'], th[0,:,0], axis=1) / (det * wtot)
      data_3d['acc_th_pmag'] = -np.gradient(det * data_3d['pmag'], th[0,:,0], axis=1) / (det * wtot)
      data_3d['acc_th_prad'] = -np.gradient(det * data_3d['prad'], th[0,:,0], axis=1) / (det * wtot)
      data_3d['acc_th_tens'] = (np.gradient(det * br * b_th, r[0,0,:], axis=2) \
          + np.gradient(det * bth * b_th, th[0,:,0], axis=1)) / (det * wtot)
      data_3d['acc_th_visc'] = \
          -(np.gradient(det * (fradr * u_th + data_3d['ur'] * frad_th), r[0,0,:], axis=2) \
          + np.gradient(det * (fradth * u_th + data_3d['uth'] * frad_th), th[0,:,0], axis=1)) \
          / (det * wtot)
      data_3d['acc_th_cent'] = 0.5 * dth_g_phph * data_3d['uph'] ** 2
      data_3d['acc_th_gr'] = 0.5 * (dth_g_tt * data_3d['ut'] ** 2 + dth_g_rr * data_3d['ur'] ** 2 \
          + dth_g_thth * data_3d['uth'] ** 2 + 2.0 * dth_g_tr * data_3d['ut'] * data_3d['ur'] \
          + 2.0 * dth_g_tth * data_3d['ut'] * data_3d['uth'] \
          + 2.0 * dth_g_tph * data_3d['ut'] * data_3d['uph'] \
          + 2.0 * dth_g_rth * data_3d['ur'] * data_3d['uth'] \
          + 2.0 * dth_g_rph * data_3d['ur'] * data_3d['uph'] \
          + 2.0 * dth_g_thph * data_3d['uth'] * data_3d['uph'])
      temp = dth_g_tt * gtt + 2.0 * dth_g_tr * gtr + 2.0 * dth_g_tth * gtth \
          + 2.0 * dth_g_tph * gtph + dth_g_rr * grr + 2.0 * dth_g_rth * grth \
          + 2.0 * dth_g_rph * grph + dth_g_thth * gthth + 2.0 * dth_g_thph * gthph \
          + dth_g_phph * gphph
      data_3d['acc_th_pgas_other'] = temp * data_3d['pgas'] / (2.0 * wtot)
      data_3d['acc_th_pmag_other'] = temp * data_3d['pmag'] / (2.0 * wtot)
      data_3d['acc_th_prad_other'] = temp * data_3d['prad'] / (2.0 * wtot)
      data_3d['acc_th_mag_other'] = -(dth_g_tt * bt ** 2 + 2.0 * dth_g_tr * bt * br \
          + 2.0 * dth_g_tth * bt * bth + 2.0 * dth_g_tph * bt * bph + dth_g_rr * br ** 2 \
          + 2.0 * dth_g_rth * br * bth + 2.0 * dth_g_rph * br * bph + dth_g_thth * bth ** 2 \
          + 2.0 * dth_g_thph * bth * bph + dth_g_phph * bph ** 2) / (2.0 * wtot)
      data_3d['acc_th_rad_other'] = dth_g_tt * fradt * data_3d['ut'] \
          + dth_g_tr * fradt * data_3d['ur'] + dth_g_tth * fradt * data_3d['uth'] \
          + dth_g_tph * fradt * data_3d['uph'] + dth_g_rr * fradr * data_3d['ur'] \
          + dth_g_rth * fradr * data_3d['uth'] + dth_g_rph * fradr * data_3d['uph'] \
          + dth_g_thth * fradth * data_3d['uth'] + dth_g_thph * fradth * data_3d['uph'] \
          + dth_g_phph * fradph * data_3d['uph']
      data_3d['acc_th_rad_other'] += dth_g_tt * data_3d['ut'] * fradt \
          + 2.0 * dth_g_tr * data_3d['ut'] * fradr + 2.0 * dth_g_tth * data_3d['ut'] * fradth \
          + 2.0 * dth_g_tph * data_3d['ut'] * fradph + dth_g_rr * data_3d['ur'] * fradr \
          + 2.0 * dth_g_rth * data_3d['ur'] * fradth + 2.0 * dth_g_rph * data_3d['ur'] * fradph \
          + dth_g_thth * data_3d['uth'] * fradth + 2.0 * dth_g_thph * data_3d['uth'] * fradph \
          + dth_g_phph * data_3d['uph'] * fradph
      data_3d['acc_th_rad_other'] /= 2.0 * wtot

      # Calculate average velocities
      uaver = 0.0
      uaveth = 0.0
      uaveph = \
          np.sum(data_3d['uph'] * data_3d['rho'], axis=0) / np.sum(data_3d['rho'], axis=0)[None,:,:]
      uavet = (-g_tph * uaveph - np.sqrt((g_tph * uaveph) ** 2 \
          - g_tt * (g_phph * uaveph ** 2 + 1.0))) / g_tt
      uave_t = g_tt * uavet + g_tph * uaveph
      uave_r = g_tr * uavet + g_rph * uaveph
      uave_ph = g_tph * uavet + g_phph * uaveph

      # Calculate transformation to fluid frame
      ft_tave = uavet
      fr_tave = 0.0
      fth_tave = 0.0
      fph_tave = uaveph
      ft_phave = 1.0
      fr_phave = 0.0
      fth_phave = 0.0
      fph_phave = -uave_t / uave_ph
      norm = np.sqrt(g_tt + 2.0 * g_tph * fph_phave + g_phph * fph_phave ** 2)
      ft_phave /= norm
      fph_phave /= norm
      ft_rave = 1.0
      fr_rave = (uave_ph * g_tt * uave_ph - uave_ph * g_tph * uave_t - uave_t * g_tph * uave_ph \
          + uave_t * g_phph * uave_t) / (uave_r * g_tph * uave_ph - uave_r * g_phph * uave_t \
          - uave_ph * g_tr * uave_ph + uave_ph * g_rph * uave_t)
      fth_rave = 0.0
      fph_rave = (uave_r * g_tph * uave_t - uave_r * g_tt * uave_ph - uave_t * g_rph * uave_t \
          + uave_t * g_tr * uave_ph) / (uave_r * g_tph * uave_ph - uave_r * g_phph * uave_t \
          - uave_ph * g_tr * uave_ph + uave_ph * g_rph * uave_t)
      norm = np.sqrt(g_tt + 2.0 * g_tr * fr_rave + 2.0 * g_tph * fph_rave + g_rr * fr_rave ** 2 \
          + 2.0 * g_rph * fr_rave * fph_rave + g_phph * fph_rave ** 2)
      ft_rave /= norm
      fr_rave /= norm
      fph_rave /= norm
      levi_civita_t = fr_phave * fth_tave * fph_rave + fph_phave * fr_tave * fth_rave \
          + fth_phave * fph_tave * fr_rave - fr_phave * fph_tave * fth_rave \
          - fth_phave * fr_tave * fph_rave - fph_phave * fth_tave * fr_rave
      levi_civita_r = ft_phave * fph_tave * fth_rave + fth_phave * ft_tave * fph_rave \
          + fph_phave * fth_tave * ft_rave - ft_phave * fth_tave * fph_rave \
          - fph_phave * ft_tave * fth_rave - fth_phave * fph_tave * ft_rave
      levi_civita_th = ft_phave * fr_tave * fph_rave + fph_phave * ft_tave * fr_rave \
          + fr_phave * fph_tave * ft_rave - ft_phave * fph_tave * fr_rave \
          - fr_phave * ft_tave * fph_rave - fph_phave * fr_tave * ft_rave
      levi_civita_ph = ft_phave * fth_tave * fr_rave + fr_phave * ft_tave * fth_rave \
          + fth_phave * fr_tave * ft_rave - ft_phave * fr_tave * fth_rave \
          - fth_phave * ft_tave * fr_rave - fr_phave * fth_tave * ft_rave
      ft_thave = sigma * sth * (gtt * levi_civita_t + gtr * levi_civita_r)
      fr_thave = sigma * sth * (gtr * levi_civita_t + grr * levi_civita_r + grph * levi_civita_ph)
      fth_thave = sigma * sth * gthth * levi_civita_th
      fph_thave = sigma * sth * (grph * levi_civita_r + gphph * levi_civita_ph)

      # Calculate covariant gas stress-energy tensor
      ttgas_tt = wgas * u_t * u_t + data_3d['pgas'] * g_tt
      ttgas_tr = wgas * u_t * u_r + data_3d['pgas'] * g_tr
      ttgas_tth = wgas * u_t * u_th
      ttgas_tph = wgas * u_t * u_ph + data_3d['pgas'] * g_tph
      ttgas_rr = wgas * u_r * u_r + data_3d['pgas'] * g_rr
      ttgas_rth = wgas * u_r * u_th
      ttgas_rph = wgas * u_r * u_ph + data_3d['pgas'] * g_rph
      ttgas_thth = wgas * u_th * u_th + data_3d['pgas'] * g_thth
      ttgas_thph = wgas * u_th * u_ph
      ttgas_phph = wgas * u_ph * u_ph + data_3d['pgas'] * g_phph

      # Calculate gas Lagrangian stress
      data_3d['Tgas_rph_f'] = ft_rave * ft_phave * ttgas_tt \
          + (ft_rave * fr_phave + fr_rave * ft_phave) * ttgas_tr \
          + (ft_rave * fth_phave + fth_rave * ft_phave) * ttgas_tth \
          + (ft_rave * fph_phave + fph_rave * ft_phave) * ttgas_tph \
          + fr_rave * fr_phave * ttgas_rr + (fr_rave * fth_phave \
          + fth_rave * fr_phave) * ttgas_rth \
          + (fr_rave * fph_phave + fph_rave * fr_phave) * ttgas_rph \
          + fth_rave * fth_phave * ttgas_thth \
          + (fth_rave * fph_phave + fph_rave * fth_phave) * ttgas_thph \
          + fph_rave * fph_phave * ttgas_phph
      data_3d['Tgas_thph_f'] = ft_thave * ft_phave * ttgas_tt \
          + (ft_thave * fr_phave + fr_thave * ft_phave) * ttgas_tr \
          + (ft_thave * fth_phave + fth_thave * ft_phave) * ttgas_tth \
          + (ft_thave * fph_phave + fph_thave * ft_phave) * ttgas_tph \
          + fr_thave * fr_phave * ttgas_rr \
          + (fr_thave * fth_phave + fth_thave * fr_phave) * ttgas_rth \
          + (fr_thave * fph_phave + fph_thave * fr_phave) * ttgas_rph \
          + fth_thave * fth_phave * ttgas_thth \
          + (fth_thave * fph_phave + fph_thave * fth_phave) * ttgas_thph \
          + fph_thave * fph_phave * ttgas_phph

      # Calculate covariant magnetic stress-energy tensor
      ttmag_tt = 2.0 * data_3d['pmag'] * u_t * u_t + data_3d['pmag'] * g_tt - b_t * b_t
      ttmag_tr = 2.0 * data_3d['pmag'] * u_t * u_r + data_3d['pmag'] * g_tr - b_t * b_r
      ttmag_tth = 2.0 * data_3d['pmag'] * u_t * u_th - b_t * b_th
      ttmag_tph = 2.0 * data_3d['pmag'] * u_t * u_ph + data_3d['pmag'] * g_tph - b_t * b_ph
      ttmag_rr = 2.0 * data_3d['pmag'] * u_r * u_r + data_3d['pmag'] * g_rr - b_r * b_r
      ttmag_rth = 2.0 * data_3d['pmag'] * u_r * u_th - b_r * b_th
      ttmag_rph = 2.0 * data_3d['pmag'] * u_r * u_ph + data_3d['pmag'] * g_rph - b_r * b_ph
      ttmag_thth = 2.0 * data_3d['pmag'] * u_th * u_th + data_3d['pmag'] * g_thth - b_th * b_th
      ttmag_thph = 2.0 * data_3d['pmag'] * u_th * u_ph - b_th * b_ph
      ttmag_phph = 2.0 * data_3d['pmag'] * u_ph * u_ph + data_3d['pmag'] * g_phph - b_ph * b_ph

      # Calculate magnetic Lagrangian stress
      data_3d['Tmag_rph_f'] = ft_rave * ft_phave * ttmag_tt \
          + (ft_rave * fr_phave + fr_rave * ft_phave) * ttmag_tr \
          + (ft_rave * fth_phave + fth_rave * ft_phave) * ttmag_tth \
          + (ft_rave * fph_phave + fph_rave * ft_phave) * ttmag_tph \
          + fr_rave * fr_phave * ttmag_rr + (fr_rave * fth_phave \
          + fth_rave * fr_phave) * ttmag_rth \
          + (fr_rave * fph_phave + fph_rave * fr_phave) * ttmag_rph \
          + fth_rave * fth_phave * ttmag_thth \
          + (fth_rave * fph_phave + fph_rave * fth_phave) * ttmag_thph \
          + fph_rave * fph_phave * ttmag_phph
      data_3d['Tmag_thph_f'] = ft_thave * ft_phave * ttmag_tt \
          + (ft_thave * fr_phave + fr_thave * ft_phave) * ttmag_tr \
          + (ft_thave * fth_phave + fth_thave * ft_phave) * ttmag_tth \
          + (ft_thave * fph_phave + fph_thave * ft_phave) * ttmag_tph \
          + fr_thave * fr_phave * ttmag_rr \
          + (fr_thave * fth_phave + fth_thave * fr_phave) * ttmag_rth \
          + (fr_thave * fph_phave + fph_thave * fr_phave) * ttmag_rph \
          + fth_thave * fth_phave * ttmag_thth \
          + (fth_thave * fph_phave + fph_thave * fth_phave) * ttmag_thph \
          + fph_thave * fph_phave * ttmag_phph

      # Calculate radiation Lagrangian stress
      data_3d['Trad_rph_f'] = ft_rave * ft_phave * ttrad_tt \
          + (ft_rave * fr_phave + fr_rave * ft_phave) * ttrad_tr \
          + (ft_rave * fth_phave + fth_rave * ft_phave) * ttrad_tth \
          + (ft_rave * fph_phave + fph_rave * ft_phave) * ttrad_tph \
          + fr_rave * fr_phave * ttrad_rr + (fr_rave * fth_phave \
          + fth_rave * fr_phave) * ttrad_rth \
          + (fr_rave * fph_phave + fph_rave * fr_phave) * ttrad_rph \
          + fth_rave * fth_phave * ttrad_thth \
          + (fth_rave * fph_phave + fph_rave * fth_phave) * ttrad_thph \
          + fph_rave * fph_phave * ttrad_phph
      data_3d['Trad_thph_f'] = ft_thave * ft_phave * ttrad_tt \
          + (ft_thave * fr_phave + fr_thave * ft_phave) * ttrad_tr \
          + (ft_thave * fth_phave + fth_thave * ft_phave) * ttrad_tth \
          + (ft_thave * fph_phave + fph_thave * ft_phave) * ttrad_tph \
          + fr_thave * fr_phave * ttrad_rr \
          + (fr_thave * fth_phave + fth_thave * fr_phave) * ttrad_rth \
          + (fr_thave * fph_phave + fph_thave * fr_phave) * ttrad_rph \
          + fth_thave * fth_phave * ttrad_thth \
          + (fth_thave * fph_phave + fph_thave * fth_phave) * ttrad_thph \
          + fph_thave * fph_phave * ttrad_phph

    # Average quantities in azimuth
    if frame_n == 0:
      data_2d = {}
    for quantity in quantities_to_average:
      data_2d[quantity] = np.mean(data_3d[quantity], axis=0)

    # Calculate post-averaging derived quantities
    data_2d['ugas'] = data_2d['pgas'] / (gamma_adi - 1.0)
    data_2d['umag'] = data_2d['pmag']
    data_2d['urad'] = 3.0 * data_2d['prad']

    # Save results
    data_out = {}
    data_out['a'] = a
    data_out['rf'] = rf
    data_out['r'] = r[0,0,:]
    data_out['thf'] = thf
    data_out['th'] = th[0,:,0]
    data_out['phf'] = phf
    data_out['ph'] = ph[:,0,0]
    for quantity in quantities_to_save:
      data_out[quantity] = data_2d[quantity]
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
  args = parser.parse_args()
  main(**vars(args))
