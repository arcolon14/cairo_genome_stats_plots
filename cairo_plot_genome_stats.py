#!/usr/bin/env python3
import sys, os, math, argparse
import cairo
import numpy as np
from IPython.display import SVG, display, Image

#
# Command line options
#
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--chroms', required=True, help='Chromosome order list')
    p.add_argument('--fai', required=True, help='Genome FASTA index')
    p.add_argument('--reps-tsv', required=True, help='Repeat window tally table')
    p.add_argument('--prot-tsv', required=True, help='Protein window tally table')
    p.add_argument('--out-dir', required=False, default='.', help='Output directory')
    p.add_argument('--name', required=True, help='Name of the dataset')
    p.add_argument('--img-height', required=False, default=500, type=int, help='Image height in pixels')
    p.add_argument('--img-width', required=False, default=500, type=int, help='Image width in pixels')
    p.add_argument('--img-format', required=False, default='pdf', help='Image output format')
    p.add_argument('--scale', required=False, default=1_000_000, type=float, help='Scale chromosome lengths to this value')
    p.add_argument('--step', required=False, default=5_000_000, type=float, help='Steps for size tick marks')
    p.add_argument('--min-len', required=False, default=1_000_000, type=float, help='Minimum length of chromosomes')
    # Check inputs
    args = p.parse_args()
    assert args.scale >= args.min_len
    assert os.path.exists(args.chroms)
    assert os.path.exists(args.fai)
    assert os.path.exists(args.reps_tsv)
    assert os.path.exists(args.prot_tsv)
    assert os.path.exists(args.out_dir)
    assert args.img_format in ['pdf', 'svg']
    args.out_dir = args.out_dir.rstrip('/')
    return args

# Load the chromosome order list
def read_chromosomes(chrom_f):
    os.path.exists(chrom_f)
    chrom_order = list()
    for line in open(chrom_f, 'r'):
        if line.startswith('#'):
            continue
        chrom = line.strip('\n').split('\t')
        assert len(chrom) == 1
        chrom_order.append(chrom[0])
    return chrom_order

#
# Load Fasta index
#

## Chromosome class
class Chromosome:
    def __init__(self, name, length):
        assert type(length) is int
        self.name = name
        self.len  = length
    def __str__(self):
        return f'{self.name} {self.len}'

def load_fai(fai_f, chromosome_order, min_len=1_000_000):
    assert os.path.exists(fai_f)
    assert type(chromosome_order) is list
    ## Loop over FAI and extract
    chromosomes = dict()
    assert os.path.exists(fai_f), f'Error: {fai_f} not found.'
    for line in open(fai_f, 'r'):
        if line.startswith('#'):
            continue
        fields = line.strip('\n').split('\t')
        name = fields[0]
        length = int(fields[1])
        if length < min_len:
            continue
        if name not in chromosome_order:
            continue
        chrom = Chromosome(name, length)
        chromosomes[name] = chrom
    return chromosomes

#
# Load the window proportions file
#

## Window statistic class
class WindowStat:
    def __init__(self, chromosome, bp=0.0, value=0.0):
        assert type(bp) in [int, float]
        assert type(value) is float
        self.chr = chromosome
        self.bp  = bp
        self.val = value
    def __str__(self):
        return f'{self.chr} {self.bp} {self.val}'

# All the columns in the file are:
# Chr  Window  Count  AdjFreq
def load_window_stats_file(win_f, chromosomes):
    assert os.path.exists(win_f)
    assert type(chromosomes) is dict
    assert isinstance(list(chromosomes.values())[0], Chromosome)

    ## Initialize the output dictionary based on the read chromosomes
    wins_dict = { chrom : [] for chrom in chromosomes }
    win_vals = list()
    mean_vals = None

    ## Parse the windows tsv
    for line in open(win_f, 'r'):
        if line.startswith('Chr'):
            continue
        fields = line.strip('\n').split('\t')
        chrom = fields[0]
        bp = float(fields[1])
        val = float(fields[3])
        if bp > chromosomes[chrom].len:
            continue
        window_stat = WindowStat(chrom, bp, val)
        if chrom not in wins_dict:
            continue
        wins_dict[chrom].append(window_stat)
        win_vals.append(val)

    mean_vals = np.mean(win_vals)

    return wins_dict, mean_vals

#
# Colors for the chromosome elements
# This includes chromosome lines, tick marks, and labels
class ChromColors:
    def __init__(self, item):
        cols_chrom = {
            'ticks'  : [192.0/255.0, 192.0/255.0, 192.0/255.0],
            'fill'   : [127.0/255.0, 127.0/255.0, 127.0/255.0],
            'line'   : [125.0/255.0, 125.0/255.0, 125.0/255.0],
            'text'   : [ 25.0/255.0,  25.0/255.0,  25.0/255.0],
            'border' : [ 50.0/255.0,  50.0/255.0,  50.0/255.0]
        }
        color  = cols_chrom[item]
        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.a = 1
    def __str__(self):
        return f'R: {self.r:.2f}, G: {self.g:.2f}, B: {self.b:.2f}, A: {self.a:.2f}'


# Other colors
# check https://www.colorhexa.com/

colors = []
# colors.append((  0.0/255.0,  80.0/255.0, 242.0/255.0)) # 0050f2 Blue
# colors.append((255.0/255.0, 255.0/255.0, 255.0/255.0)) # ffffff White
# colors.append((255.0/255.0,   0.0/255.0,   0.0/255.0)) # ff0000 Red

# # Viridis
# colors.append(( 25.0/255.0,  25.0/255.0, 112.0/255.0)) # #191970 Indigo
# colors.append((  0.0/255.0, 105.0/255.0, 062.0/255.0)) # #00693e Green
# colors.append((255.0/255.0, 168.0/255.0,  18.0/255.0)) # #ffa812 Yellow

# Mango
# colors.append((  0.0/255.0, 105.0/255.0, 062.0/255.0)) # #00693e Green
# colors.append((255.0/255.0, 168.0/255.0,  18.0/255.0)) # #ffa812 Yellow
# colors.append((178.0/255.0,  34.0/255.0,  34.0/255.0)) # ff0000 Red

# Magma
# colors.append((  0.0/255.0,   0.0/255.0, 139.0/255.0)) # #191970 Indigo
# colors.append((255.0/255.0,  40.0/255.0,   0.0/255.0)) # ff0000 Red
# colors.append((252.0/255.0, 247.0/255.0,  94.0/255.0)) # #ffa812 Yellow

colors.append((  0.0/255.0,  80.0/255.0, 242.0/255.0)) # 0050f2 Blue
colors.append((252.0/255.0, 247.0/255.0,  94.0/255.0)) # #ffa812 Yellow
colors.append((255.0/255.0,   0.0/255.0,   0.0/255.0)) # ff0000 Red


def scale_sigmoid_color_mean(mean, val):
    #
    # Plot the values using a Gompertz curve: y(t) = ae^(-be(^-ct))
    #   https://en.wikipedia.org/wiki/Gompertz_function
    #
    # The X-axis scales from 0 to 10, Y-axis from 0 to 1; we have shifted
    # the curve to the right by subtracting 2 from val; we will shit it more
    # to line the sigmoid curve up with the mean value.
    #
    val  = math.fabs(val)
    val  = val * 10
    mean = mean * 10 / 2
    a    = 1
    b    = 1
    c    = 0.5
    val  = a * math.exp(-1 * b * (math.exp(-1 * c * (val - 2 - mean))))
    return val

def three_color_gradient(rgb1, rgb2, rgb3, mean, alpha):
    (r1, g1, b1) = rgb1
    (r2, g2, b2) = rgb2
    (r3, g3, b3) = rgb3

    alpha = scale_sigmoid_color_mean(mean, alpha)
    if (alpha < mean):
        scaled_alpha = float(alpha) / mean
        r = ((1.0 - scaled_alpha) * r1) + (scaled_alpha * r2)
        g = ((1.0 - scaled_alpha) * g1) + (scaled_alpha * g2)
        b = ((1.0 - scaled_alpha) * b1) + (scaled_alpha * b2)
    else:
        scaled_alpha = float(alpha - mean) / (1.0 - mean)
        r = (scaled_alpha * r3) + ((1.0 - scaled_alpha) * r2)
        g = (scaled_alpha * g3) + ((1.0 - scaled_alpha) * g2)
        b = (scaled_alpha * b3) + ((1.0 - scaled_alpha) * b2)
    return (r, g, b)

# Set PyCairo environment
#
# Class to set the settings to the image output
class Image:
    def __init__(self, height=500, width=500, font_size=18, img_type='pdf', offset=25, edge=15):
        assert offset > edge
        # General Plot info
        self.height = height
        self.width  = width
        self.font   = font_size
        self.type   = img_type
        self.max_y  = edge
        self.min_y  = height-edge
        self.min_x  = edge
        self.max_x  = width-edge
        # Position for the objects
        self.min_chr = offset
        self.max_chr = self.max_x
        self.min_tck = self.max_y
        self.max_tck = height-offset
        # Position for labels
        self.len_lab = height-edge
        self.chr_lab = offset-edge
    def scale_bp_to_pix(self, pos_bp, max_bp):
        scaled_x = ((pos_bp/max_bp)*(self.max_chr - self.min_chr)) + self.min_chr
        return scaled_x
    def cairo_context(self, plot_out_dir):
        if self.type == 'pdf':
            surface = cairo.PDFSurface(plot_out_dir, self.width, self.height)
            context = cairo.Context(surface)
            return surface, context
        elif self.type == 'svg':
            surface = cairo.SVGSurface(plot_out_dir, self.width, self.height)
            context = cairo.Context(surface)
            return surface, context
        else:
            sys.exit("The only supported formats are \'PDF\' and \'SVG\'.")
    # Display image
    def show_svg(self, plot_out_dir):
        if self.type != 'svg':
            sys.exit('Image can only be displayed in SVG format')
        display(SVG(filename=file))

#
# Plot gridlines
#
def plot_gridlines(chromosomes, image, context, scale=1_000_000, step=5_000_000):
    assert type(chromosomes) is dict
    assert isinstance(list(chromosomes.values())[0], Chromosome)
    assert isinstance(image, Image)
    # Plot the gridlines
    max_bp = max([ chromosomes[chrom].len for chrom in chromosomes ])
    assert max_bp > step
    grid = list(range(0,(max_bp+step),step))
    max_grd = max(grid)
    for bp in grid:
        # Add gridlines
        y1 = image.min_tck
        y2 = image.max_tck+(image.max_y/4)
        x = image.scale_bp_to_pix(bp, max_grd)
        context.set_dash([6.0, 3.0])
        context.set_line_cap(cairo.LINE_CAP_BUTT)
        context.move_to(x, y1)
        context.line_to(x, y2)
        ticks_col = ChromColors('ticks')
        context.set_source_rgb(ticks_col.r, ticks_col.g, ticks_col.b)
        context.set_line_width(0.75)
        context.stroke()
        # Add labels
        label = f'{int(bp//scale)}M'
        label_height = context.text_extents(label)[3]
        label_width  = context.text_extents(label)[2]
        txt_col = ChromColors('text')
        lab_x = x-(label_width/2)
        lab_y = image.len_lab+(label_height)
        context.move_to(lab_x, lab_y)
        context.set_source_rgb(txt_col.r, txt_col.g, txt_col.b)
        context.show_text(label)
    return max_grd

#
# Process the chromosomes
#
def process_chromosomes(chromosomes, chrom_order, reps_dict, image, context, max_grd, mean_reps, scale=1_000_000, step=5_000_000):
    assert type(chromosomes) is dict
    assert isinstance(list(chromosomes.values())[0], Chromosome)
    assert type(reps_dict) is dict
    assert isinstance(image, Image)
    assert len(chromosomes) == len(chrom_order)

    # Plot the chromosomes and values
    chr_step = (image.max_tck-image.min_tck)/len(chromosomes)
    min_step = chr_step/4
    y = image.min_tck

    ## Loop over the chromosomes
    # for chromosome in sorted(chromosomes, key=lambda c: chromosomes[c].len, reverse=True):
    for chromosome in chrom_order:
        assert chromosome in chromosomes
        # Define the boundaries of the chromosome
        bp = chromosomes[chromosome].len
        x1 = image.min_chr
        x2 = image.scale_bp_to_pix(bp, max_grd)
        y1 = y+(1*min_step)
        y2 = y+(3*min_step)

        # Extract the windows for the current chromosome
        windows = reps_dict.get(chromosome, [])
        if len(windows) > 0:
            for window in windows:
                assert isinstance(window, WindowStat)
                if window.bp > bp:
                    continue
                x = image.scale_bp_to_pix(window.bp, max_grd)
                (r, g, b) = three_color_gradient(colors[0], colors[1], colors[2], mean_reps, window.val)
                context.set_dash([])
                context.move_to(x, y1)
                context.line_to(x, y2)
                context.set_source_rgb(r, g, b)
                context.stroke()

        # Add the Chromosome len boxes
        context.set_dash([])
        context.move_to(x1, y1)
        context.line_to(x2, y1)
        context.line_to(x2, y2)
        context.line_to(x1, y2)
        context.close_path()
        context.set_line_width(1.0)
        col = ChromColors('border')
        context.set_source_rgb(col.r, col.g, col.b)
        # context.stroke_preserve()
        context.stroke()
        # col = ChromColors('fill')
        # context.set_source_rgb(col.r, col.g, col.b)
        # context.fill()

        # Add the labels
        label = str(chromosomes[chromosome].name)
        label_height = context.text_extents(label)[3]
        label_width  = context.text_extents(label)[2]
        lab_x = image.chr_lab-(label_width/2)
        lab_y = (y+(2*min_step))+(label_height/2)
        txt_col = ChromColors('text')
        context.move_to(lab_x, lab_y)
        context.set_source_rgb(txt_col.r, txt_col.g, txt_col.b)
        context.show_text(label)
        y += chr_step

#
# Draw the Scale
#
def draw_scale(image, context, mean_reps):
    assert isinstance(image, Image)
    # Boundaries
    x1 = image.max_x*0.945
    x2 = image.max_x*0.995
    y1 = image.max_tck*0.795
    y2 = image.max_tck*0.995
    # Loop over the color space
    s=0.005
    for p in np.arange(0,(1+s),s):
        (r, g, b) = three_color_gradient(colors[0], colors[1], colors[2], mean_reps, p)
        key_h = y2-y1
        yp = y2-(key_h*p)
        context.set_dash([])
        context.move_to(x1, yp)
        context.line_to(x2, yp)
        context.set_source_rgb(r, g, b)
        context.stroke()
    # Outer box
    context.set_dash([])
    context.move_to(x1, y1)
    context.line_to(x2, y1)
    context.line_to(x2, y2)
    context.line_to(x1, y2)
    context.close_path()
    context.set_line_width(1.0)
    col = ChromColors('border')
    context.set_source_rgb(col.r, col.g, col.b)
    # context.stroke_preserve()
    context.stroke()
    # col = ChromColors('fill')
    # context.set_source_rgb(col.r, col.g, col.b)
    # context.fill()

    #
    # Add labels
    #
    # Label 1
    lab1 = f'{1.0}'
    label_height = context.text_extents(lab1)[3]
    label_width  = context.text_extents(lab1)[2]
    lab_x = x1-(label_width*1.25)
    lab_y = y1+(label_height/2)
    txt_col = ChromColors('text')
    context.move_to(lab_x, lab_y)
    context.set_source_rgb(txt_col.r, txt_col.g, txt_col.b)
    context.show_text(lab1)
    # Label 2
    lab2 = f'{0.0}'
    label_height = context.text_extents(lab2)[3]
    label_width  = context.text_extents(lab2)[2]
    lab_x = x1-(label_width*1.25)
    lab_y = y2+(label_height/2)
    txt_col = ChromColors('text')
    context.move_to(lab_x, lab_y)
    context.set_source_rgb(txt_col.r, txt_col.g, txt_col.b)
    context.show_text(lab2)

# Add title to the figure
def draw_title(image, context, title):
    assert isinstance(image, Image)
    x = image.min_chr
    y = image.max_y
    # Adjust height
    title_height = context.text_extents(title)[3]
    y = y-(title_height/2)
    txt_col = ChromColors('text')
    context.move_to(x,y)
    context.set_source_rgb(txt_col.r, txt_col.g, txt_col.b)
    context.show_text(title)


# Draw a figure
def draw_genome_stats(outf, chromosomes, chrom_order, win_val_dict, mean_val, name, height=500, width=500, scale=1_000_000, step=5_000_000, img_type='pdf'):
    # Set an image object global variable
    image = Image(height=height, width=width, img_type=img_type)
    surface, context = image.cairo_context(outf)
    # Plot gridlines
    max_grd = plot_gridlines(chromosomes, image, context, scale, step)
    # Process the chromosomes
    process_chromosomes(chromosomes, chrom_order, win_val_dict, image, context, max_grd, mean_val, scale, step)
    # Plot the scale
    draw_scale(image, context, mean_val)
    # Add title
    draw_title(image, context, name)


def main():
    args = parse_args()
    chrom_order = read_chromosomes(args.chroms)
    # Load fai
    chromosomes = load_fai(args.fai, chrom_order, args.min_len)
    # Get Max Chr size
    max_bp = max([ chromosomes[chrom].len for chrom in chromosomes ])
    # Load repeats
    reps_dict, mean_reps = load_window_stats_file(args.reps_tsv, chromosomes)
    # Load genes
    gene_dict, mean_gene = load_window_stats_file(args.prot_tsv, chromosomes)

    # ============
    # Draw Repeats
    # ============

    # Create an output file
    outf = f'{args.out_dir}/repeat_stats.{args.img_format}'
    name = f'{args.name} Repeats'
    # Draw
    draw_genome_stats(outf, chromosomes, chrom_order, reps_dict, mean_reps, name, height=args.img_height, width=args.img_width, scale=args.scale, step=args.step, img_type=args.img_format)

    # =============
    # Draw Proteins
    # =============

    # Create an output file
    outf = f'{args.out_dir}/genes_stats.{args.img_format}'
    name = f'{args.name} Genes'
    # Draw
    draw_genome_stats(outf, chromosomes, chrom_order, gene_dict, mean_gene, name, height=args.img_height, width=args.img_width, scale=args.scale, step=args.step, img_type=args.img_format)

# Run Code
if __name__ == '__main__':
    main()
