# Plot heatmaps of genome-wide stats

Plots heatmaps of the density of repeat and protein-coding elements along the chromosomes.

## Binning the elements

```sh
$ ./bin_genome_stats.py -h
usage: bin_genome_stats.py [-h] --fai FAI --rep-tsv REP_TSV --prot-gff
                           PROT_GFF [--out-dir OUT_DIR] [--win-size WIN_SIZE]
                           [--win-step WIN_STEP] [--min-len MIN_LEN]

options:
  -h, --help           show this help message and exit
  --fai FAI            Genome FASTA index
  --rep-tsv REP_TSV    Repeat position table
  --prot-gff PROT_GFF  Protein position table/GFF file
  --out-dir OUT_DIR    Output directory
  --win-size WIN_SIZE  Size of windows (in bp)
  --win-step WIN_STEP  Step of windows (in bp)
  --min-len MIN_LEN    Minimum length of chromosomes
```

### Inputs

Genome FASTA index, as generated by `samtools faidx <fasta.fa>` ([docs](http://www.htslib.org/doc/samtools-faidx.html)).

```sh
chr01  73382502  7          60  61
chr02  65036697  74605558   60  61
chr03  63127771  140726207  60  61
chr06  60469010  204906115  60  61
chr04  56595772  266382949  60  61
chr08  56204502  323921991  60  61
chr07  56023676  381063242  60  61
chr05  55627136  438020653  60  61
chr09  55004135  494574915  60  61
chr10  53367998  550495793  60  61
```

Protein-coding gene annotation in GTF/GFF [format](http://useast.ensembl.org/info/website/upload/gff.html).

```sh
##gtf-version  3
chr01          LiftOn  gene        29011   29320   .    +  .  gene_id  "gene-1"
chr01          LiftOn  transcript  29011   29320   .    +  .  gene_id  "gene-1"
chr01          LiftOn  exon        29011   29320   .    +  .  gene_id  "gene-1"
chr01          LiftOn  CDS         29011   29320   .    +  0  gene_id  "gene-1"
chr01          LiftOn  gene        101162  101701  421  +  .  gene_id  "gene-2"
chr01          LiftOn  transcript  101162  101701  421  +  .  gene_id  "gene-2"
chr01          LiftOn  exon        101162  101701  421  +  0  gene_id  "gene-2"
chr01          LiftOn  CDS         101162  101701  421  +  0  gene_id  "gene-2"
chr01          LiftOn  gene        113722  114014  .    -  .  gene_id  "gene-3"
```

Table of the location of repeat elements.

```sh
#Chr   Start  End   RepName    RepClass
chr01  18     223   Unknown    Unknown
chr01  234    675   Unknown    Unknown
chr01  747    1095  Unknown    Unknown
chr01  1452   1664  Unknown    Unknown
chr01  1665   1971  Unknown    Unknown
chr01  2069   5825  Satellite  Satellite
chr01  5873   6072  Unknown    Unknown
chr01  6073   6316  Unknown    Unknown
chr01  6317   7509  Unknown    Unknown
```

TODO: How to generate this from the RepeatMasker [output table](https://www.repeatmasker.org/webrepeatmaskerhelp.html).

## Making the plots

Requires [PyCairo](https://pycairo.readthedocs.io/en/latest/).

```sh
$ ./cairo_plot_genome_stats.py -h
usage: cairo_plot_genome_stats.py [-h] --chroms CHROMS
                                  --fai FAI --reps-tsv
                                  REPS_TSV --prot-tsv
                                  PROT_TSV
                                  [--out-dir OUT_DIR]
                                  --name NAME
                                  [--img-height IMG_HEIGHT]
                                  [--img-width IMG_WIDTH]
                                  [--img-format IMG_FORMAT]
                                  [--scale SCALE]
                                  [--step STEP]
                                  [--min-len MIN_LEN]

options:
  -h, --help            show this help message and exit
  --chroms CHROMS       Chromosome order list
  --fai FAI             Genome FASTA index
  --reps-tsv REPS_TSV   Repeat window tally table
  --prot-tsv PROT_TSV   Protein window tally table
  --out-dir OUT_DIR     Output directory
  --name NAME           Name of the dataset
  --img-height IMG_HEIGHT
                        Image height in pixels
  --img-width IMG_WIDTH
                        Image width in pixels
  --img-format IMG_FORMAT
                        Image output format
  --scale SCALE         Scale chromosome lengths to this
                        value
  --step STEP           Steps for size tick marks
  --min-len MIN_LEN     Minimum length of chromosomes
```

### Inputs

Chromosome order list: Text file containing the chromosome IDs (one per line) in the order they will be plotted.

```sh
chr01
chr02
chr03
chr06
chr04
chr08
chr07
chr05
chr09
chr10
```

## Some examples

Figure 2B and 2C ([link](https://academic.oup.com/view-large/figure/397327756/msad029f3.tif)) from [Rivera-Colon et al. 2023](https://doi.org/10.1093/molbev/msad029).

Figure 3 ([link](https://academic.oup.com/view-large/figure/499727840/jkae267f3.jpg)) from [Rayamajhi et al. 2024](https://doi.org/10.1093/g3journal/jkae267).

TODO: Add some images.

## Author

Angel G. Rivera-Colon  
Institute of Ecology and Evolution  
University of Oregon
