#!/usr/bin/env python3
import sys, os, argparse, datetime, gzip
import numpy as np
# import pandas as pd

# Some constants
PROG = sys.argv[0].split('/')[-1]
MIN_CHR_LEN = 1_000_000
WIN_SIZE = 100_000
WIN_STEP = 10_000

def parse_args(prog=PROG):
    '''Set and verify command line options.'''
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--fai', required=True, 
                   help='(str) Path to genome index in FAI format.')
    # TODO: set this to a single input in BED format.
    # TODO: Generate a basename for the outputs.
    # p.add_argument('--rep-tsv', required=True, help='Repeat position table')
    # p.add_argument('--prot-gff', required=True, help='Protein position table/GFF file')
    p.add_argument('-o', '--out-dir', required=False, default='.',
                   help='(str) Path to output directory [default=.].')
    p.add_argument('-s', '--win-size', required=False, default=WIN_SIZE, type=float,
                   help=f'(int/float) Size of windows in bp [default {WIN_SIZE}].')
    p.add_argument('-t', '--win-step', required=False, default=WIN_STEP, type=float,
                   help=f'(int/float) Step of windows in bp [default {WIN_STEP}].')
    p.add_argument('-m', '--min-len', required=False, default=MIN_CHR_LEN,
                   type=float, help=f'(int/float) Minimum chromosome size in bp [default {MIN_CHR_LEN}]')
    # Check inputs
    args = p.parse_args()
    assert args.win_size > args.win_step
    assert args.min_len > args.win_size
    assert os.path.exists(args.fai)
    # assert os.path.exists(args.rep_tsv)
    # assert os.path.exists(args.prot_gff)
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
    return args

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
            windows = calculate_chr_window_intervals(seq_len, window_size, window_step)
            genome_window_intervals[seq_id] = windows
            # Set the lengths for future logs
            seq_lens[seq_id] = seq_len
    # Report some stats on the windows
    print(f'\nRead {n_seqs:,} total records from the FAI file.\nGenerated window intervals for {len(genome_window_intervals):,} chromosomes/scaffolds:', flush=True)
    for chr in genome_window_intervals:
        print(f'    {chr}: {seq_lens[chr]:,} bp; {len(genome_window_intervals[chr]):,} windows')

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


#
# Calculate the window intervals for a given Chromosome length
#
def calculate_chr_window_intervals(chr_len, window_size=WIN_SIZE, window_step=WIN_STEP):
    assert window_size >= window_step
    assert window_size > 0
    windows = list()
    window_start = 0
    window_end = window_size
    while window_end < (chr_len+window_step):
        windows.append((window_start, window_end))
        window_start += window_step
        window_end += window_step
    return windows

# Tally the windows in the dataframe
# Dataframa must have two columns, Start bp and End bp
def tally_df_windows(coordinates_df, chrom_len, chrom_id, win_size=100_000, step=10_000):
    assert isinstance(coordinates_df, pd.DataFrame)
    assert len(coordinates_df.columns) == 2, 'Dataframe must have two columns, start and end coordinates.'
    assert chrom_len >= win_size
    assert step <= win_size
    # Main output
    win_cnt_df = pd.DataFrame(columns=('Chr', 'Window','Count'))
    i = 0
    # Loop over the windows
    for win_sta in np.arange(0, chrom_len+step, step):
        # Window properties
        win_end = win_sta+win_size
        win_mid = win_sta+(win_size//2)
        if win_mid >= chrom_len:
            continue
        # Slice the dataframe
        win_elements = coordinates_df[(coordinates_df.iloc[:,0] >= win_sta) & (coordinates_df.iloc[:,1] < win_end)]
        new_row = [str(chrom_id), win_mid, win_elements.shape[0]]
        win_cnt_df.loc[i] = new_row
        i+=1
    return win_cnt_df

# Tally the windows for the repeat dattaframe
def tally_repeat_windows(repeat_df, chrom_len, chrom_id, win_size=100_000, step=10_000):
    assert isinstance(repeat_df, pd.DataFrame)
    assert chrom_len >= win_size
    assert step <= win_size
    # Main output
    win_cnt_dict = dict()
    reps_bin_df = pd.DataFrame(columns=('Chr', 'Window','Count','DNA','SINE','LINE','LTR','Other'))
    i = 0
    # Loop over the windows
    for win_sta in np.arange(0, chrom_len+step, step):
        # Window properties
        win_end = win_sta+win_size
        win_mid = win_sta+(win_size//2)
        if win_mid >= chrom_len:
            continue
        # Slice the dataframe
        win_elements = repeat_df[(repeat_df['Start'] >= win_sta) & (repeat_df['End'] < win_end)]
        new_row = [str(chrom_id), win_mid, win_elements.shape[0]]
        counts = win_elements['RepType'].value_counts()
        for t in ['DNA','SINE','LINE','LTR','Other']:
            val = None
            try:
                val = counts[t]
            except:
                val = 0.0
            new_row.append(val)
        reps_bin_df.loc[i] = new_row
        i+=1
    return reps_bin_df
 
# Returns a dataframe for each object
def get_chr_stats(curr_chr, fai, prot, reps, win_size=100_000, win_step=10_000):
    # Filter the fasta index and obtain some length variable
    curr_fai  = fai[fai['id'] == curr_chr].copy()
    chr_len = curr_fai.iloc[0]['len']

    # Filter the protein file
    curr_prot = prot[prot['Chr'] == curr_chr].copy()
    curr_prot = curr_prot.sort_values(by=['Start'])
    coordinates_df = curr_prot[['Start','End']]
    prot_win_cnt = tally_df_windows(coordinates_df, chr_len, curr_chr, win_size, win_step)

    # Filter the repeat file
    curr_reps = reps[reps['#Chr'] == curr_chr].copy()
    curr_reps = curr_reps.sort_values(by=['Start'])
    curr_reps = curr_reps[['Start','End', 'RepType']]
    # rep_win_cnt = tally_df_windows(coordinates_df, chr_len, curr_chr, win_size, win_step)
    rep_win_cnt = tally_repeat_windows(curr_reps, chr_len, curr_chr, win_size, win_step)

    return prot_win_cnt, rep_win_cnt

# Load the FASTA index file
def load_fai(fai_f, min_len=1_000_000):
    assert os.path.exists(fai_f)
    columns = ['id', 'len', 'byte_idx', 'bp_per_line', 'byte_per_line']
    fai = pd.read_table(fai_f, header=None, names=columns, dtype={'id':str})
    ## Get the two first columns only
    fai = fai[['id','len']]
    ## Filter by minimum size
    fai = fai[fai['len'] > min_len]
    # fai = fai.astype({'id' : str})
    fai['id'] = pd.to_numeric(fai['id'])
    return fai

# Load the repeats file
def load_reps(reps_f, fai_df):
    assert os.path.exists(reps_f)
    assert isinstance(fai_df, pd.DataFrame)
    chromosomes = fai_df['id'].astype(str).tolist()
    reps = pd.read_table(reps_f, header=[0], dtype={'#Chr':str})
    reps = reps[reps['#Chr'].isin(chromosomes)]
    reps['#Chr'] = pd.to_numeric(reps['#Chr'])
    # Bin the repeat types
    reps = bin_repeat_classes(reps)
    return reps

# Load the protein coding gene file
def load_prots(prot_f, fai_df):
    assert os.path.exists(prot_f)
    assert isinstance(fai_df, pd.DataFrame)
    chromosomes = fai_df['id'].astype(str).tolist()
    prot = pd.read_table(prot_f, header=None, dtype={0:str})
    prot = prot[prot[2] == 'gene']
    prot = prot[[0,3,4,8]]
    prot.columns = ['Chr','Start','End','GeneID']
    prot = prot[prot['Chr'].isin(chromosomes)]
    prot['Chr'] = pd.to_numeric(prot['Chr'])
    return prot

# breakdown repeat classes
def bin_repeat_classes(rep_df):
    assert isinstance(rep_df, pd.DataFrame)
    rep_type_list = list()
    for repeat in rep_df['RepClass']:
        repeat = repeat.split('/')[0]
        if repeat in ['DNA', 'RC']:
            rep_type_list.append('DNA')
        elif repeat in ['SINE','SINE?']:
            rep_type_list.append('SINE')
        elif repeat in ['LINE','LINE?']:
            rep_type_list.append('LINE')
        elif repeat in ['LTR']:
            rep_type_list.append('LTR')
        else:
            rep_type_list.append('Other')
    rep_df['RepType'] = rep_type_list
    return rep_df

# Process all the chromosomes
def process_chromosome_windows(fai_df, rep_df, prot_df, outd, win_size, win_step):
    assert isinstance(fai_df, pd.DataFrame)
    assert isinstance(rep_df, pd.DataFrame)
    assert isinstance(prot_df, pd.DataFrame)
    assert os.path.exists(outd)
    # Preprate outputs
    outd = outd.rstrip('/')
    reps_tsv = f'{outd}/chr_reps_win.tsv'
    prot_tsv = f'{outd}/chr_prot_win.tsv'

    chr_i = 0
    total_prot_win_cnt = None
    total_rep_win_cnt  = None
    for chrom in fai_df['id'].sort_values().tolist():
        chr_prot_win_cnt, chr_rep_win_cnt = get_chr_stats(chrom, fai_df, prot_df, rep_df, win_size, win_step)
        # Initialize the total dataframes for the first chromosome
        if chr_i == 0:
            total_prot_win_cnt = chr_prot_win_cnt.copy()
            total_rep_win_cnt = chr_rep_win_cnt.copy()
        # Append the rest
        else:
            total_prot_win_cnt = pd.concat([total_prot_win_cnt, chr_prot_win_cnt])
            total_rep_win_cnt = pd.concat([total_rep_win_cnt, chr_rep_win_cnt])
        chr_i += 1

    # Add the frequencies and save to a file

    ## Protein File
    # total_prot_win_cnt['AdjFreq'] = total_prot_win_cnt['Count']/max(total_prot_win_cnt['Count'])
    total_prot_win_cnt['AdjFreq'] = total_prot_win_cnt['Count']/np.mean(total_prot_win_cnt['Count'])
    total_prot_win_cnt.to_csv(prot_tsv, sep='\t', index=False)
    ## Repeats (Small)
    total_rep_win_cnt_sm = total_rep_win_cnt[['Chr', 'Window','Count']].copy()
    # total_rep_win_cnt_sm['AdjFreq'] = total_rep_win_cnt_sm['Count']/max(total_rep_win_cnt_sm['Count'])
    total_rep_win_cnt_sm['AdjFreq'] = total_rep_win_cnt_sm['Count']/np.mean(total_rep_win_cnt_sm['Count'])
    total_rep_win_cnt_sm.to_csv(reps_tsv, sep='\t', index=False)
    ## Repeats (Detailed)
    reps_details_tsv = f'{outd}/chr_reps_win.detailed.tsv'
    total_rep_win_cnt['Count'] = total_rep_win_cnt['Count'] / max(total_rep_win_cnt['Count'])
    total_rep_win_cnt['DNA']   = total_rep_win_cnt['DNA']   / max(total_rep_win_cnt['DNA'])
    total_rep_win_cnt['SINE']  = total_rep_win_cnt['SINE']  / max(total_rep_win_cnt['SINE'])
    total_rep_win_cnt['LINE']  = total_rep_win_cnt['LINE']  / max(total_rep_win_cnt['LINE'])
    total_rep_win_cnt['LTR']   = total_rep_win_cnt['LTR']   / max(total_rep_win_cnt['LTR'])
    total_rep_win_cnt['Other'] = total_rep_win_cnt['Other'] / max(total_rep_win_cnt['Other'])
    total_rep_win_cnt.to_csv(reps_details_tsv, sep='\t', index=False)

def main():
    args = parse_args()
    # Initialize script
    print(f'{PROG} started on {date()} {time()}.')
    print(f'    Min Chrom Length: {int(args.min_len):,} bp')
    print(f'    Window Size: {int(args.win_size):,} bp')
    print(f'    Window Step: {int(args.win_step):,} bp', flush=True)

    # Get windows from the fai
    genome_window_intervals = set_windows_from_fai(args.fai, args.win_size, args.win_step, args.min_len)
    # # Load fai
    # fai = load_fai(args.fai, args.min_len)
    # # Load repeats  
    # reps = load_reps(args.rep_tsv, fai)
    # # Load proteins
    # prot = load_prots(args.prot_gff, fai)

    # # Process all chromosomes
    # process_chromosome_windows(fai, reps, prot, args.out_dir, args.win_size, args.win_step)

    print(f'\n{PROG} finished on {date()} {time()}.')

# Run Code
if __name__ == '__main__':
    main()
