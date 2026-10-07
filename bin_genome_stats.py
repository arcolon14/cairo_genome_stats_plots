#!/usr/bin/env python3
import sys
import os
import argparse
import datetime
import gzip
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
        args.basename = os.path.basename(args.in_bed).removesuffix('.bed')
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
    '''
    Store the coordinates and attributes of a target
    genomic window.
    '''
    def __init__(self, chromosome:str, start_bp:int|float, end_bp:int|float):
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
        self.n_bases = 0     # Number of bases covered by elements (overlapping bases counted multiple times)
        self.n_sites = 0     # Number of sites covered by the elements (overlapping bases counted once)
    def __str__(self):
        row = f'{self.wid} {self.chr} {self.sta} {self.end} {self.n_bases} {self.n_elements} {self.n_sites}'
        return row

def date():
    '''Print the current date in YYYY-MM-DD format.'''
    return datetime.datetime.now().strftime("%Y-%m-%d")

def time():
    '''Print the current time in HH:MM:SS format.'''
    return datetime.datetime.now().strftime("%H:%M:%S")

def set_windows_from_fai(fai, window_size=WIN_SIZE, window_step=WIN_STEP, min_chr_size=MIN_CHR_LEN):
    '''
    Use the genome fasta index to pre-calculate the genomic windows.
    Based on the script:
    https://github.com/adeflamingh/de_Flamingh_etal_2023_Cape_lion/blob/main/average_genomic_windows.py
    '''
    assert window_step > 0
    genome_window_intervals = dict()
    seq_lens = dict()
    n_seqs = 0
    with open(fai, encoding='utf-') as fh:
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
    for chrom in genome_window_intervals:
        print(f'    {chrom}: {seq_lens[chrom]:,} bp; {len(genome_window_intervals[chrom]):,} windows', flush=True)

    return genome_window_intervals

def init_windows_dictionary(genome_window_intervals):
    '''
    Generate a dictionary of window values and initialize with zeroes.
    '''
    windows_dict = {}
    for chrom in genome_window_intervals:
        chr_windows = genome_window_intervals[chrom]
        windows_dict.setdefault(chrom, {})
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

def merge_intervals(starts, ends):
    '''
    Merge a set of intervals into disjoint, sorted blocks. Overlapping
    and adjacent intervals are combined into a single block.
    '''
    assert len(starts) == len(ends)
    if len(starts) == 0:
        return starts, ends
    order = np.argsort(starts, kind='stable')
    starts = starts[order]
    ends = ends[order]
    # A new block begins when an interval starts past the furthest end seen so far
    running_end = np.maximum.accumulate(ends)
    new_block = np.ones(len(starts), dtype=bool)
    new_block[1:] = starts[1:] > running_end[:-1]
    block_idx = np.flatnonzero(new_block)
    block_starts = starts[block_idx]
    block_ends = np.maximum.reduceat(ends, block_idx)
    return block_starts, block_ends

def covered_sites_before(positions, block_starts, block_ends):
    '''
    For each position, count the number of sites covered by the
    (disjoint, sorted) blocks to the left of that position.
    '''
    # Total length of all the blocks before each block
    cumul_len = np.concatenate([[0], np.cumsum(block_ends-block_starts)])
    # Index of the last block starting at or before each position
    k = np.searchsorted(block_starts, positions, side='right') - 1
    kk = np.clip(k, 0, None)
    covered = cumul_len[kk] + np.minimum(positions, block_ends[kk]) - block_starts[kk]
    return np.where(k >= 0, covered, 0)

def element_bases_before(positions, starts, ends):
    '''
    For each position, sum the number of bases of all elements to the left
    of that position. Bases of overlapping elements are counted multiple times.
    '''
    starts = np.sort(starts)
    ends = np.sort(ends)
    cumul_starts = np.concatenate([[0], np.cumsum(starts)])
    cumul_ends = np.concatenate([[0], np.cumsum(ends)])
    # Elements starting before the position contribute (position - start) bases,
    # minus (position - end) bases for those that also end before the position.
    n_sta = np.searchsorted(starts, positions, side='left')
    n_end = np.searchsorted(ends, positions, side='left')
    bases = (n_sta*positions - cumul_starts[n_sta]) - (n_end*positions - cumul_ends[n_end])
    return bases

def tally_chromosome_windows(chr_windows, starts, ends):
    '''
    Tally the number of elements, and the number of sites covered by
    them, in all the windows of a given chromosome.
    '''
    assert len(starts) == len(ends)
    if len(starts) == 0:
        return chr_windows
    win_sta = np.array([ window.sta for window in chr_windows ], dtype=np.int64)
    win_end = np.array([ window.end for window in chr_windows ], dtype=np.int64)
    # Number of elements overlapping each window. An element overlaps the
    # window [win_sta, win_end) if it starts before win_end and ends after
    # win_sta. Since start < end for all elements, this equals the number of
    # elements starting before win_end minus those ending at or before win_sta.
    n_elements = (np.searchsorted(np.sort(starts), win_end, side='left') -
                  np.searchsorted(np.sort(ends), win_sta, side='right'))
    # Number of bases of each window covered by the elements
    n_bases = (element_bases_before(win_end, starts, ends) -
               element_bases_before(win_sta, starts, ends))
    # Number of sites in each window covered by the elements. Merge the elements
    # first so that sites belonging to overlapping elements are counted once.
    block_starts, block_ends = merge_intervals(starts, ends)
    n_sites = (covered_sites_before(win_end, block_starts, block_ends) -
               covered_sites_before(win_sta, block_starts, block_ends))
    # Add the tallies to the windows
    for i, window in enumerate(chr_windows):
        assert isinstance(window, GenomicWindow)
        window.n_elements = int(n_elements[i])
        window.n_bases = int(n_bases[i])
        window.n_sites = int(n_sites[i])
        assert window.n_sites <= window.n_bases
        assert window.n_sites <= (window.end - window.sta)
    return chr_windows

def extract_elements_from_input_bed(in_bed_f, genomic_windows, min_span=MIN_SPAN):
    '''
    Parse the input bed file and tally elements in the windows.
    '''
    assert os.path.exists(in_bed_f)
    assert isinstance(genomic_windows, dict)
    print(f'\nParsing input BED file:\n    {in_bed_f}', flush=True)

    # Prepare outputs
    seen_records = 0
    kept_records = 0
    # Coordinates of the kept elements, per chromosome
    chr_elements = { chrom : ([], []) for chrom in genomic_windows }
    with open(in_bed_f, encoding='utf-8') as fh:
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
            if end_bp < start_bp:
                sys.exit(f'Error: Mis-formatted BED. End coordinate (column 3) must be larger than start coordinate (column 2) (line {i+1}).')
            seen_records += 1
            # Skip entries that are not in the window chromosomes
            if chromosome not in genomic_windows:
                continue
            # Skip entries that are under the desired length (span)
            if (end_bp-start_bp) < min_span:
                continue
            # Error if the range is not within the chromosome
            if end_bp > genomic_windows[chromosome][-1].end:
                sys.exit(f'Error: Range {start_bp} to {end_bp} not within the range of sequence {chromosome} (line {i+1}).')
            # Add this entry to the elements of the chromosome
            chr_elements[chromosome][0].append(start_bp)
            chr_elements[chromosome][1].append(end_bp)
            kept_records += 1
    print(f'\n    Read {seen_records:,} records from input BED file.\n    Kept a total of {kept_records:,} records.', flush=True)

    # Tally the elements in the windows of each chromosome
    for chrom in genomic_windows:
        starts = np.array(chr_elements[chrom][0], dtype=np.int64)
        ends = np.array(chr_elements[chrom][1], dtype=np.int64)
        genomic_windows[chrom] = tally_chromosome_windows(genomic_windows[chrom], starts, ends)
    return genomic_windows

def generate_genome_wide_averages(genomic_windows)->dict:
    '''
    Iterate over all the populated windows and generate the
    genome-wide of the proportion of bases covered and number
    of elements per-window.
    '''
    # Store all the averages for both number of elements and proportions
    genome_averages = {}
    n_elements = [] # Average number of elements per window
    bp_prop = []    # Average proportion of bases of the element per window
    # Loop over all the windows...
    for chrom in genomic_windows:
        for window in genomic_windows[chrom]:
            assert isinstance(window, GenomicWindow)
            # The number of elements just gets appended as is. It is just a tally
            n_elements.append(window.n_elements)
            # For the proportion, it has to be calculated based on the number of bases
            # covered and the length of the window.
            # TODO: mean of non-zero elements???
            window_len = window.end - window.sta
            n_sites = window.n_sites
            assert n_sites <= window_len, f'{window}'
            prop_mean = n_sites/window_len
            bp_prop.append(prop_mean)
    # Calculate the stats for the number of elements
    genome_averages.setdefault('n_elements', {})
    genome_averages['n_elements']['mean'] = np.mean(n_elements)
    genome_averages['n_elements']['median'] = np.median(n_elements)
    genome_averages['n_elements']['std'] = 0
    if len(n_elements) > 0:
        genome_averages['n_elements']['std'] = np.std(n_elements)

    # Calculate the stats for the proportion of bases
    genome_averages.setdefault('bp_prop', {})
    genome_averages['bp_prop']['mean'] = np.mean(bp_prop)
    genome_averages['bp_prop']['median'] = np.median(bp_prop)
    genome_averages['bp_prop']['std'] = 0
    if len(n_elements) > 0:
        genome_averages['bp_prop']['std'] = np.std(bp_prop)

    # Report to log.
    print(f'''
    Genome-wide average number of elements in a window:
        Mean:   {genome_averages['n_elements']['mean']:,.6g}
        Median: {genome_averages['n_elements']['median']:,.6g}
        StDev:  {genome_averages['n_elements']['std']:,.6g}
    Genome-wide average proportion of bases in a window:
        Mean:   {genome_averages['bp_prop']['mean']:,.6g}
        Median: {genome_averages['bp_prop']['median']:,.6g}
        StDev:  {genome_averages['bp_prop']['std']:,.6g}''',
    flush=True)
    return genome_averages

def process_windows_output(genomic_windows, output_dir, basename):
    '''
    Process the populated genomic windows, calculate averages, and
    export outputs.
    '''
    # Generate the output file
    outf = f'{output_dir}/{basename}.binned_genome_stats.tsv'
    print(f'\nGenerating output to:\n    {outf}', flush=True)
    # Get the genome-wide averages of the proportion and number of
    # elements per window.
    genome_averages = generate_genome_wide_averages(genomic_windows)
    element_mean = genome_averages['n_elements']['mean']
    prop_mean = genome_averages['bp_prop']['mean']

    # Generate the output file handle
    with open(outf, 'w', encoding='utf-8') as fh:
        header = ['#Chrom', 'StartBP', 'EndBP', 'MidBP',
                  'ElementsN', 'ElementsAdj',
                  'PropSites', 'PropSitesAdj']
        header = '\t'.join(header)
        fh.write(f'{header}\n')
        # Loop over the windows and write to file
        for chrom in genomic_windows:
            for window in genomic_windows[chrom]:
                assert isinstance(window, GenomicWindow)
                # Start with the four standard positional entries
                row = f'{window.chr}\t{window.sta}\t{window.end}\t{window.mid}'
                # Then process the rest and add to the list
                # Number of elements per window
                # Adjust based on the mean as a log2 enrichment, with a
                # pseudocount of one element to handle empty windows
                elements_adj = np.log2((window.n_elements+1)/(element_mean+1))
                row += f'\t{window.n_elements}\t{elements_adj:0.8g}'

                # Proportion of elements in window
                window_len = window.end-window.sta
                prop_elements = window.n_sites/window_len
                # Adjust based on the mean as a log2 enrichment, with a
                # pseudocount of one site (1/window length) to handle empty windows
                prop_pseudo = 1/window_len
                prop_adj = np.log2((prop_elements+prop_pseudo)/(prop_mean+prop_pseudo))
                row += f'\t{prop_elements:0.8g}\t{prop_adj:0.8g}'
                fh.write(f'{row}\n')

def main():
    print(f'{PROG} started on {date()} {time()}.')
    args = parse_args()
    # Initialize script
    print(f'    Min Chrom Length: {int(args.min_len):,} bp')
    print(f'    Window Size: {int(args.win_size):,} bp')
    print(f'    Window Step: {int(args.win_step):,} bp', flush=True)
    if args.min_span > 1:
        print(f'    Min Size of Input Element in BED: {int(args.min_span):,} bp',
              flush=True)

    # Get windows from the fai
    genome_window_intervals = set_windows_from_fai(args.fai,
                                                   args.win_size,
                                                   args.win_step,
                                                   args.min_len)

    # Process the input bed
    genome_window_intervals = extract_elements_from_input_bed(args.in_bed,
                                                              genome_window_intervals,
                                                              args.min_span)

    # Generate a new output file
    process_windows_output(genome_window_intervals, args.out_dir, args.basename)

    # Done!
    print(f'\n{PROG} finished on {date()} {time()}.')


# Run Code
if __name__ == '__main__':
    main()
