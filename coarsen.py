#! /usr/bin/env python3

"""
Script for coarsening AthenaK data.

Usage:
[python3] coarsen.py \
    <input_file_base> <output_file_base> \
    <frame_min> <frame_max> <frame_stride> \
    <factor> \
    [--mpi]

<input_file_base> should refer to standard AthenaK .bin data dumps that use some
form of Cartesian coordinates with volume given by the product of the cell
widths. Files should have the form <input_file_base>.{05d}.bin.

The files <output_file_base>.{05d}.bin will be overwritten. They will contain
the coarsened data in the same format.

File numbers from <frame_min> to <frame_max>, inclusive with a stride of
<frame_stride>, will be processed.

The coarsening will be done by averaging blocks of <factor>^3 cells. <factor>
must divide all 3 MeshBlock dimensions.

Options:
  --mpi: flag indicating MPI should be used to distribute work
  --verbose (-v): flag indicating detailed progress should updates should be given

Averaging is done in file quantities, and so may not be exactly conservative.

If given data in coordinates with nonconstant metric determinant, this script
will produce improperly weighted volume averages.

This script is optimized for using a minimal amount of memory.
"""

# Python standard modules
import argparse
import os
import struct

# Numerical modules
import numpy as np

# Main function
def main(**kwargs):

  # Parameters - inputs
  input_file_base = kwargs['input_file_base']
  output_file_base = kwargs['output_file_base']
  frame_min = kwargs['frame_min']
  frame_max = kwargs['frame_max']
  frame_stride = kwargs['frame_stride']
  factor = kwargs['factor']
  use_mpi = kwargs['mpi']
  verbose = kwargs['verbose']

  # Check for files being overwritten
  input_file = '{0}.{1:05d}.bin'.format(input_file_base, frame_min)
  output_file = '{0}.{1:05d}.bin'.format(output_file_base, frame_min)
  if os.path.isfile(input_file):
    if os.path.isfile(output_file):
      if os.path.samefile(input_file, output_file):
        raise RuntimeError('Attempting to overwrite data; coarsening in place not supported.')
  else:
    raise RuntimeError('Input file {0} not found.'.format(input_file))

  # Check for sensible coarsening factor
  if factor <= 1:
    raise RuntimeError('Must choose coarsening factor greater than 1.')

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
    output_file = '{0}.{1:05d}.bin'.format(output_file_base, frame)

    # Read non-varying input data on first pass
    if frame_n == 0:
      with open(input_file, 'rb') as f:

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
        variable_names = line[12:].split()
        line = f.readline().decode('ascii')
        if line[:16] != '  header offset=':
          raise RuntimeError('Could not read header offset.')
        header_offset = int(line[16:])

        # Process header metadata
        if location_size not in (4, 8):
          raise RuntimeError('Only 4- and 8-byte floating-point types supported for location data.')
        location_format = 'f' if location_size == 4 else 'd'
        if variable_size not in (4, 8):
          raise RuntimeError('Only 4- and 8-byte floating-point types supported for cell data.')
        variable_format = 'f' if variable_size == 4 else 'd'
        num_variables = len(variable_names)
        start_of_data = f.tell() + header_offset

        # Read data from first block
        f.seek(start_of_data, 0)
        block_indices_old = struct.unpack('@6i', f.read(24))
        n_x_old = block_indices_old[1] - block_indices_old[0] + 1
        n_y_old = block_indices_old[3] - block_indices_old[2] + 1
        n_z_old = block_indices_old[5] - block_indices_old[4] + 1

        # Process data from first block
        cells_per_block_old = n_z_old * n_y_old * n_x_old
        block_cell_format_old = '=' + str(num_variables * cells_per_block_old) + variable_format
        variable_data_size_old = cells_per_block_old * variable_size
        block_size_old = 40 + 6 * location_size + num_variables * variable_data_size_old

    # Calculate coarsened block sizes on first pass
    if frame_n == 0:
      block_indices_new = np.copy(block_indices_old)
      if n_x_old == 1:
        n_x_new = 1
        coarsen_x = False
      else:
        if n_x_old % factor != 0:
          raise RuntimeError('Coarsening factor does not divide x-size of blocks.')
        n_x_new = n_x_old // factor
        block_indices_new[1] = block_indices_new[0] + n_x_new - 1
        coarsen_x = True
      if n_y_old == 1:
        n_y_new = 1
        coarsen_y = False
      else:
        if n_y_old % factor != 0:
          raise RuntimeError('Coarsening factor does not divide y-size of blocks.')
        n_y_new = n_y_old // factor
        block_indices_new[3] = block_indices_new[2] + n_y_new - 1
        coarsen_y = True
      if n_z_old == 1:
        n_z_new = 1
        coarsen_z = False
      else:
        if n_z_old % factor != 0:
          raise RuntimeError('Coarsening factor does not divide z-size of blocks.')
        n_z_new = n_z_old // factor
        block_indices_new[5] = block_indices_new[4] + n_z_new - 1
        coarsen_z = True
      cells_per_block_new = n_z_new * n_y_new * n_x_new
      block_cell_format_new = '=' + str(num_variables * cells_per_block_new) + variable_format

    # Open old and new files for reading and writing
    with open(input_file, 'rb') as f_old:
      with open(output_file, 'wb') as f_new:

        # Get input file size
        f_old.seek(0, 2)
        file_size_old = f_old.tell()
        f_old.seek(0, 0)

        # Read input file header metadata
        for n in range(8):
          next(f_old)
        line = f_old.readline().decode('ascii')
        if line[:16] != '  header offset=':
          raise RuntimeError('Could not read header offset.')
        header_offset = int(line[16:])

        # Process input file header metadata
        start_of_data = f_old.tell() + header_offset
        if (file_size_old - start_of_data) % block_size_old != 0:
          raise RuntimeError("File has unexpected layout.")
        num_blocks = (file_size_old - start_of_data) // block_size_old

        # Read and write header metadata and metadata
        f_old.seek(0, 0)
        f_new.write(f_old.read(start_of_data))

        # Go through blocks
        for block_n in range(num_blocks):

          # Report progress
          if verbose:
            if not use_mpi:
              print('File {0} / {1}, block {2} / {3}'\
                  .format(frame_n + 1, len(frames), block_n + 1, num_blocks))
            elif rank == 0:
              print('Rank 0: file {0} / {1}, block {2} / {3}'\
                  .format(frame_n + 1, num_frames_local, block_n + 1, num_blocks))

          # Read and write grid structure data
          f_old.seek(24, 1)
          f_new.write(struct.pack('@6i', *block_indices_new))
          f_new.write(f_old.read(16))

          # Read and write coordinate data
          f_new.write(f_old.read(6 * location_size))

          # Read cell data
          cell_data = f_old.read(num_variables * variable_data_size_old)
          cell_data = struct.unpack(block_cell_format_old, cell_data)
          cell_data = np.array(cell_data).reshape(num_variables, n_z_old, n_y_old, n_x_old)

          # Coarsen cell data
          if coarsen_x:
            cell_data = \
                np.mean(cell_data.reshape(num_variables, n_z_old, n_y_old, n_x_new, factor), axis=4)
          if coarsen_y:
            cell_data = \
                np.mean(cell_data.reshape(num_variables, n_z_old, n_y_new, factor, n_x_new), axis=3)
          if coarsen_z:
            cell_data = \
                np.mean(cell_data.reshape(num_variables, n_z_new, factor, n_y_new, n_x_new), axis=2)

          # Write cell data
          f_new.write(struct.pack(block_cell_format_new, *cell_data.flatten()))

# Process inputs and execute main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file_base', help='base name of AthenaK .bin files to coarsen')
  parser.add_argument('output_file_base', help='base name of new .bin files to write')
  parser.add_argument('frame_min', type=int, help='initial file number to process')
  parser.add_argument('frame_max', type=int, help='final file number to process')
  parser.add_argument('frame_stride', type=int, help='stride in file numbers to process')
  parser.add_argument('factor', type=int, help='coarsening stencil size in each dimension')
  parser.add_argument('--mpi', action='store_true', \
      help='flag indicating MPI should be used to distribute work')
  parser.add_argument('-v', '--verbose', action='store_true', \
      help='flag indicating detailed progress should updates should be given')
  args = parser.parse_args()
  main(**vars(args))
