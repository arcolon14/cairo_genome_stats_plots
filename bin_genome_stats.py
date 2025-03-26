#!/usr/bin/env python3
import sys, os, argparse, datetime, gzip
import numpy as np

# Some constants
PROG = sys.argv[0].split('/')[-1]
MIN_CHR_LEN = 1_000_000
WIN_SIZE = 100_000
WIN_STEP = 50_000
MIN_SPAN = 1

def parse_args(prog=PROG):
    '''Set and verify command line options.'''
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--fai', required=True, 
                   help='(str) Path to genome index in FAI format.')
    p.add_argument('-b', '--in-bed', required=True,
                   help='(str) Path to input BED file describing the windows to be tallied.')
    p.add_argument('-n', '--basename', required=False, default=None,
                   help='(str) Basename of output files [default=BED Basename].')
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
    assert args.win_size >= args.win_step
    assert args.min_len > args.win_size
    assert os.path.exists(args.fai)
    assert os.path.exists(args.in_bed)
    assert os.path.exists(args.out_dir)
    args.out_dir = args.out_dir.rstrip('/')
    # Proceess the basename if not provided
    if args.basename is None:
        args.basename = args.in_bed.rstrip('.bed')
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
        self.sta = int(start_bp)
        self.end = int(end_bp)
        self.mid = int(start_bp+((end_bp-start_bp)/2))
        # Window ID; <chrom ID>:<position>, e.g., chr01:123456
        self.wid = f'{chromosome}:{self.mid}'
        # Initialize the tallies
        self.n_elements = 0  # Number of elements in the window
        self.n_bases = 0     # Number of bases covered by elements
        self.sites = set()   # Set of the sites coveraged by the elements (to handle overlaps)
    def __str__(self):
        row = f'{self.wid} {self.chr} {self.sta} {self.end} {self.n_bases} {self.n_elements} {len(self.sites)}'
        return row
    def find_overlapping_bps(self, target_start, target_end):
        assert type(target_start) in {int, float}
        assert type(target_end) in {int, float}
        assert target_end > target_start
        target = set(range(int(target_start), int(target_end)))
        sites = set(range(int(self.sta), int(self.end)))
        overlap = sites.intersection(target)
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
        if window_end > chr_len:
            window_end = chr_len
        genomic_window = GenomicWindow(chr_id, window_start, window_end)
        windows.append(genomic_window)
        window_start += window_step
        window_end += window_step
    return windows


def binary_search_windows(chr_windows, target_bp):
    '''
    Use a binary search to find the interval where to iterate over
    the desired chromosome windows.
    '''
    low = 0
    high = len(chr_windows) - 1
    mid = 0
    # First, confirm that the target element is within the desired 
    # range. If not, return None and raise error in next step.
    if target_bp > chr_windows[-1].end:
        return None
    while low <= high:
        # This divides the chromosome windows into two halves
        # based on the midpoint of the number of windows. These 
        # "halves" become smaller as we proportionally slice 
        # down the chromosome into smaller and smaller chunks.
        mid = (high + low) // 2
        # Find the window at the midpoint, this will allow us
        # to compare our target BP position.
        curr_window = chr_windows[mid]
        assert isinstance(curr_window, GenomicWindow)
        # print(low, mid, high, curr_window)
        # If the target BP is larger than the midpoint,
        # ignore the first half of the interval.
        if curr_window.sta <= target_bp:
            low = mid + 1
        # Instead, if the target is smaller than the midpoint
        # position, ignore the second half of the interval
        elif curr_window.sta > target_bp:
            high = mid - 1
        else:
            return mid
    # The previous loop ends one window past the target, so 
    # return back one and return.
    mid -= 1
    return mid

def add_bed_record_to_windows(bed_chr, bed_start, bed_end, genomic_windows):
    '''
    Add a given BED record to the genomic windows dictionary.
    '''
    assert type(genomic_windows) is dict
    # First, work only on the windows of the target chromosome
    chr_windows = genomic_windows[bed_chr]
    # We will traverse the chromosome windows using a binary search
    # operation (or at least, binary search-like).
    # These will provide the start index we are going to use to 
    # iterate over the windows.
    start_idx = binary_search_windows(chr_windows, bed_start)
    # print(start_idx, chr_windows[start_idx])
    if start_idx is None:
        # Error if the range is not within the chromosome
        sys.exit(f'Error: Range {bed_start} to {bed_end} not within the range of sequence {bed_chr}.')
    # start_idx = 0
    # Iterate over the chromosome windows starting from the selected 
    # point. End once the windows move past the range of the record.
    for win_i in range(start_idx, len(chr_windows)-1):
        curr_window = chr_windows[win_i]
        assert isinstance(curr_window, GenomicWindow)
        # If the current window is before the target record, keep moving 
        # up the windows. This shouldn't happen, since we did the binary 
        # search above, but good as a safety net.
        if curr_window.end < bed_start:
            continue
        # If the current window is after the target record, stop.
        if curr_window.sta > bed_end:
            break
        # Determine the range of the overlap.
        overlap = curr_window.find_overlapping_bps(bed_start, bed_end)
        # Add this to the tally
        if len(overlap) > 0:
            curr_window.n_bases += len(overlap)
            curr_window.n_elements += 1
            curr_window.sites.update(list(overlap))
        # Add it back to the original, genome-wide object
        genomic_windows[bed_chr][win_i] = curr_window
    # Return the original, genome-wide windows object with the values
    # of the present record added.
    return genomic_windows


def extract_elements_from_input_bed(in_bed_f, genomic_windows, min_span=MIN_SPAN):
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
            # Add the present record to the windows dictionary
            genomic_windows = add_bed_record_to_windows(chromosome, start_bp, end_bp, genomic_windows)
            # Add this entry to the window dictionary
            kept_records += 1
    print(f'\n    Read {seen_records:,} records from input BED file.\n    Kept a total of {kept_records:,} records.', flush=True)
    return genomic_windows


def generate_genome_wide_averages(genomic_windows):
    '''
    Iterate over all the populated windows and generate the
    genome-wide of the proportion of bases covered and number
    of elements per-window.
    '''
    n_elements_mean = list() # Average number of elements per window
    bp_prop_mean = list()    # Average proportion of bases of the element per window
    # Loop over all the windows...
    for chr in genomic_windows:
        for window in genomic_windows[chr]:
            assert isinstance(window, GenomicWindow)
            # The number of elements just gets appended as is. It is just a tally
            n_elements_mean.append(window.n_elements)
            # For the proportion, it has to be calculated based on the number of bases covered and the length ofg the window.
            # TODO: mean of non-zero elements???
            window_len = window.end - window.sta
            n_sites = len(window.sites)
            assert n_sites <= window_len, f'{window}'
            prop_mean = n_sites/window_len
            bp_prop_mean.append(prop_mean)
    # Now calculate the averages with the populated lists
    n_elements_mean = np.mean(n_elements_mean)
    bp_prop_mean = np.mean(bp_prop_mean)
    # Report to log.
    print(f'\n    Genome-wide mean number of elements in a window: {n_elements_mean:,.6g}')
    print(f'    Genome-wide mean proportion of target bases in a window: {bp_prop_mean:,.6g}', flush=True)
    return n_elements_mean, bp_prop_mean


def process_windows_output(genomic_windows, output_dir, basename):
    '''
    Process the populated genomic windows, calculate averages, and
    export outputs.
    '''
    # Generate the output file
    outf = f'{output_dir}/{basename}.binned_genome_stats.tsv'
    print(f'\nGenerating output to:\n    {outf}', flush=True)
    # Get the genome-wide mean of the proportion and number of 
    # elements per window.
    n_elements_mean, bp_prop_mean = generate_genome_wide_averages(genomic_windows)
    # Generate the output file handle
    with open(outf, 'w') as fh:
        header = ['#Chrom', 'StartBP', 'EndBP', 'MidBP',
                  'ElementsN', 'ElementsAdj', 'PropSites', 'PropSitesAdj']
        header = '\t'.join(header)
        fh.write(f'{header}\n')
        # Loop over the windows and write to file
        for chr in genomic_windows:
            for window in genomic_windows[chr]:
                assert isinstance(window, GenomicWindow)
                # Start with the four standard positional entries
                row = f'{window.chr}\t{window.sta}\t{window.end}\t{window.mid}'
                # Then process the rest and add to the list
                # Number of elements per window
                elements_adj = window.n_elements/n_elements_mean
                row += f'\t{window.n_elements}\t{elements_adj:0.8g}\t'
                # Proportion of elements in window
                prop_elements = len(window.sites)/(window.end-window.sta)
                prop_elements_adj = prop_elements/bp_prop_mean
                row += f'{prop_elements:0.8g}\t{prop_elements_adj:0.8g}'
                fh.write(f'{row}\n')


def main():
    print(f'{PROG} started on {date()} {time()}.')
    args = parse_args()
    # Initialize script
    print(f'    Min Chrom Length: {int(args.min_len):,} bp')
    print(f'    Window Size: {int(args.win_size):,} bp')
    print(f'    Window Step: {int(args.win_step):,} bp', flush=True)
    if args.min_span > 1: 
        print(f'    Min Size of Input Element in BED: {int(args.min_span):,} bp', flush=True)

    # Get windows from the fai
    genome_window_intervals = set_windows_from_fai(args.fai, args.win_size, args.win_step, args.min_len)

    # Process the input bed
    genome_window_intervals = extract_elements_from_input_bed(args.in_bed, genome_window_intervals, args.min_span)

    # Generate a new output file
    process_windows_output(genome_window_intervals, args.out_dir, args.basename)

    # Done!
    print(f'\n{PROG} finished on {date()} {time()}.')


# Run Code
if __name__ == '__main__':
    main()
