#! /usr/bin/env python3

"""
Script for axisymmetrizing data in SKS NumPy format.

Usage:
[python3] axisymmetrize_sks.py \
    <input_file_base> <output_file_base> \
    <frame_min> <frame_max> <frame_stride>

<input_file_base> should refer to .npz files as produced by cks_to_sks.py or
cks_to_sks_lowmem.py, for example. Files should have the form
<input_file_base>.{05d}.npz.

The files <output_file_base>.{05d}.npz will be overwritten. They will contain
the axisymmetrized data.

File numbers from <frame_min> to <frame_max>, inclusive with a stride of
<frame_stride>, will be processed.
"""

# Python standard modules
import argparse

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

  # Parameters - array names
  names_constants = ('a', 'gamma_adi')
  names_coords_2d = ('rf', 'r', 'thf', 'th')
  names_coords_all = names_coords_2d + ('phf', 'ph')

  # Calculate which frames to process
  frames = range(frame_min, frame_max + 1, frame_stride)

  # Go through files
  for frame_n, frame in enumerate(frames):

    # Calculate file names
    input_file = '{0}.{1:05d}.npz'.format(input_file_base, frame)
    output_file = '{0}.{1:05d}.npz'.format(output_file_base, frame)

    # Read input data
    data_3d = np.load(input_file)

    # Prepare dictionary to hold results
    data_2d = {}

    # Extract constants
    for name in names_constants:
      data_2d[name] = data_3d[name]

    # Extract coordinates
    for name in names_coords_2d:
      data_2d[name] = data_3d[name]

    # Axisymmetrize data
    names = [key for key in data_3d.keys()]
    for name in names:
      if name in names_constants + names_coords_all:
        continue
      vals_3d = np.array(data_3d[name])
      num_nan_3d = np.sum(np.isnan(vals_3d))
      data_2d[name] = np.nanmean(vals_3d, axis=0)
      num_nan_2d = np.sum(np.isnan(data_2d[name]))
      if num_nan_3d > 0:
        print('Frame {0}, {1}: {2} -> {3} NaN values'.format(frame, name, num_nan_3d, num_nan_2d))

    # Write output data
    np.savez(output_file, **data_2d)

# Process inputs and execute main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file_base', help='base name of .npz files to read')
  parser.add_argument('output_file_base', help='base name of .npz files to write')
  parser.add_argument('frame_min', type=int, help='initial file number to process')
  parser.add_argument('frame_max', type=int, help='final file number to process')
  parser.add_argument('frame_stride', type=int, help='stride in file numbers to process')
  args = parser.parse_args()
  main(**vars(args))
