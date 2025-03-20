#!/usr/bin/env python3
import sys, os, argparse, datetime, gzip
import numpy as np
# import pandas as pd

# Some constants
PROG = sys.argv[0].split('/')[-1]
MIN_CHR_LEN = 1_000_000
WIN_SIZE = 100_000
WIN_STEP = 10_000
MIN_SPAN = 1
NAME = 'binned_genome_stats'

def parse_args(prog=PROG):
    '''Set and verify command line options.'''
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--fai', required=True, 
                   help='(str) Path to genome index in FAI format.')
    p.add_argument('-b', '--in-bed', required=True,
                   help='(str) Path to input BED file describing the windows to be tallied.')
    p.add_argument('-n', '--basename', required=False, default=NAME,
                   help=f'(str) Basename of output files [default={NAME}].')
    # TODO: set this to a single input in BED format.
    # TODO: Generate a basename for the outputs.
    # p.add_argument('--rep-tsv', required=True, help='Repeat position table')
    # p.add_argument('--prot-gff', required=True, help='Protein position table/GFF file')
    p.add_argument('-o', '--out-dir', required=False, default='.',
                   help='(str) Path to output directory [default=.].')
    p.add_argument('-s', '--win-size', required=False, default=WIN_SIZE, type=float,
                   help=f'(int/float) Size of windows in bp [default {WIN_SIZE:,}].')
    p.add_argument('-t', '--win-step', required=False, default=WIN_STEP, type=float,
                   help=f'(int/float) Step of windows in bp [default {WIN_STEP:,}].')
    p.add_argument('-m', '--min-len', required=False, default=MIN_CHR_LEN,
                   type=float, help=f'(int/float) Minimum chromosome size in bp [default {MIN_CHR_LEN:,}]')
    p.add_argument('-p', '--min-span', required=False, type=float, default=MIN_SPAN,
                   help=f'(int/float) Minimum genomic span in bp required to keep an element from the input BED file [default={MIN_SPAN:,}].')
    # Check inputs
    args = p.parse_args()
    assert args.win_size > args.win_step
    assert args.min_len > args.win_size
    assert os.path.exists(args.fai)
    assert os.path.exists(args.in_bed)
    assert os.path.exists(args.out_dir)
    args.out_dir = args.out_dir.rstrip('/')
    # Check the lengths
    if not args.win_size > 0:
        sys.exit(f"Error: size of windows ({args.win_size}) must be > 0.")
    if not args.win_step > 0:
        sys.exit(f"Error: step of windows ({args.win_step}) must be > 0.")
    if not args.min_len > 0:
        sys.exit(f"Error: Min chromosome length ({args.min_len}) must be > 0.")
    if not args.win_size >= args.win_step:
        sys.exit(f"Error: Window size ({args.win_size}) must be >= than window step ({args.win_step}).")
    if not args.min_span > 0:
        sys.exit(f"Error: Min genomic window span ({args.min_span}) must be > 0.")
    return args


class GenomicWindow():
    def __init__(self, chromosome, start_bp, end_bp):
        # Check the input coordinates
        assert type(start_bp) in {int, float}
        assert type(end_bp) in {int, float}
        assert end_bp > start_bp
        # Define base attributes
        self.chr = chromosome
        self.sta = start_bp
        self.end = end_bp
        self.mid = (end_bp-start_bp)+start_bp
        # Window ID; <chrom ID>:<position>, e.g., chr01:123456
        self.wid = f'{chromosome}:{int(self.mid)}'
        # Initialize the tallies
        self.n_elements = 0  # Number of elements in the window
        self.n_bases = 0     # Number of bases covered by elements
    def __str__(self):
        row = f'{self.wid} {self.chr} {self.sta} {self.end} {self.n_bases} {self.n_elements}'
        return row
    def find_overlapping_bps(self, target_start, target_end):
        assert type(target_start) in {int, float}
        assert type(target_end) in {int, float}
        assert target_end > target_start
        target = set(range(int(target_start), int(target_end)))
        window = set(range(int(self.sta), int(self.end)))
        overlap = window.intersection(target)
        return overlap


def date():
    '''Print the current date in YYYY-MM-DD format.'''
    return datetime.datetime.now().strftime("%Y-%m-%d")

def time():
    '''Print the current time in HH:MM:SS format.'''
    return datetime.datetime.now().strftime("%H:%M:%S")

def set_windows_from_fai(fai, window_size=WIN_SIZE, window_step=WIN_STEP, min_chr_size=MIN_CHR_LEN):
    '''
    Use the genome fasta index to pre-calculate the genomic windows.
    Based on the script: https://github.com/adeflamingh/de_Flamingh_etal_2023_Cape_lion/blob/main/average_genomic_windows.py
    '''
    assert window_step > 0
    genome_window_intervals = dict()
    seq_lens = dict()
    n_seqs = 0
    with open(fai) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.strip('\n').split('\t')
            if not len(fields) >= 2:
                sys.exit('Error: FAI must have at least two columns, seq_id<tab>seq_len')
            seq_id = fields[0]
            if not fields[1].isnumeric():
                sys.exit('Error: column two of the FAI must be a number with the length of the sequence.')
            seq_len = int(fields[1])
            n_seqs += 1
            if seq_len < min_chr_size:
                continue
            # Prepate the windows
            windows = calculate_chr_window_intervals(seq_id, seq_len, window_size, window_step)
            genome_window_intervals[seq_id] = windows
            # Set the lengths for future logs
            seq_lens[seq_id] = seq_len
    # Report some stats on the windows
    print(f'\nRead {n_seqs:,} total records from the FAI file.\n\nGenerated window intervals for {len(genome_window_intervals):,} chromosomes/scaffolds:', flush=True)
    for chr in genome_window_intervals:
        print(f'    {chr}: {seq_lens[chr]:,} bp; {len(genome_window_intervals[chr]):,} windows', flush=True)

    return genome_window_intervals

def init_windows_dictionary(genome_window_intervals):
    '''
    Generate a dictionary of window values and initialize with zeroes.
    '''
    windows_dict = dict()
    for chrom in genome_window_intervals:
        chr_windows = genome_window_intervals[chrom]
        windows_dict.setdefault(chrom, dict())
        for window in chr_windows:
            assert len(window) == 2
            window_sta = window[0]
            window_end = window[1]
            window_mid = int(window_sta + ((window_end - window_sta)/2))
            deflt = [0, 0]
            # TODO: A default class for the windows?
            windows_dict[chrom].setdefault(window_mid, deflt)
    return windows_dict


def calculate_chr_window_intervals(chr_id, chr_len, window_size=WIN_SIZE, window_step=WIN_STEP):
    '''
    Calculate the window intervals for a given Chromosome length
    '''
    assert window_size >= window_step
    assert window_size > 0
    windows = list()
    window_start = 0
    window_end = window_size
    while window_end < (chr_len+window_step):
        genomic_window = GenomicWindow(chr_id, window_start, window_end)
        windows.append(genomic_window)
        window_start += window_step
        window_end += window_step
    return windows


def parse_bed(in_bed_f, genomic_windows, min_span=MIN_SPAN):
    '''
    Parse the input bed file and tally elements in the windows.
    '''
    assert os.path.exists(in_bed_f)
    assert type(genomic_windows) is dict
    print(f'\nParsing input BED file:\n    {in_bed_f}', flush=True)

    # Prepare outputs
    seen_records = 0
    kept_records = 0
    with open(in_bed_f) as fh:
        for i, line in enumerate(fh):
            line = line.strip('\n')
            # Skip comments and empty lines
            if line.startswith('#') or len(line) == 0:
                continue
            fields = line.split('\t')
            # Check for BED integrity (must be at least 3 columns)
            if len(fields) < 3:
                sys.exit(f'Error: Mis-formatted BED file. Must contain at least 3 columns (line {i+1}).')
            # Set the three needed fields in the BED, the rest are optional and can be ignored.
            chromosome = fields[0]
            start_bp = fields[1]
            end_bp = fields[2]
            # Columns 2 and 3 must be numeric coordinates
            if not start_bp.isnumeric() or not end_bp.isnumeric():
                sys.exit(f'Error: Mis-formatted BED. Columns 2 and 3 must be numeric (line {i+1}).')
            start_bp = int(start_bp)
            end_bp = int(end_bp)
            # End column must be larger than start column
            # BED is 0-based, inclusive for start, exclusive for end, so
            # even 1-bp intervals should follow this convention.
            if not end_bp > start_bp:
                sys.exit(f'Error: Mis-formatted BED. End coordinate (column 3) must be larger than start coordinate (column 2) (line {i+1}).')
            seen_records += 1
            # Skip entries that are not in the window chromosomes
            if chromosome not in genomic_windows:
                continue
            # Skip entries that are under the desired length (span)
            if (end_bp-start_bp) < min_span:
                continue
            # Add the record to the windows dictionary
            for win_i in range(len(genomic_windows[chromosome])):
                curr_window = genomic_windows[chromosome][win_i]
                assert isinstance(curr_window, GenomicWindow)
                # If the current window is before the target record, keep moving up the windows
                if curr_window.end < start_bp:
                    continue
                # If the current window is after the target record, stop.
                if curr_window.sta > end_bp:
                    break
                # Determine the range of the overlap.
                overlap = curr_window.find_overlapping_bps(start_bp, end_bp)
                print(len(overlap))
                # TODO: Add this to the tally




            # Add this entry to the window dictionary
            kept_records += 1

            # print([chromosome, start_bp, end_bp])
            # if i > 20:
            #     break
    print(f'    Read {seen_records:,} records from input BED file.\n    Kept a total of {kept_records:,} records.', flush=True)

 











def main():
    args = parse_args()
    # Initialize script
    print(f'{PROG} started on {date()} {time()}.')
    print(f'    Min Chrom Length: {int(args.min_len):,} bp')
    print(f'    Window Size: {int(args.win_size):,} bp')
    print(f'    Window Step: {int(args.win_step):,} bp', flush=True)

    # Get windows from the fai
    genome_window_intervals = set_windows_from_fai(args.fai, args.win_size, args.win_step, args.min_len)

    # Process the input bed
    parse_bed(args.in_bed, genome_window_intervals, args.min_span)
    # print(genome_window_intervals)



    print(f'\n{PROG} finished on {date()} {time()}.')

# Run Code
if __name__ == '__main__':
    main()
