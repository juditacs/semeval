from sys import argv
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pylab as pl




def parse_args():
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('--legend-pos', type=str, default='upper right',
                   help='legend position, "upper left" or "upper right"'
                   'should do the trick')
    p.add_argument('--xlabel', type=str, default='threshold', help='label of axis X')
    p.add_argument('--ylabel', type=str, default='F score',
                  help='label of axis Y')
    p.add_argument('--x-col', type=int, default=0,
                  help='column number (index starting from 0) of '
                  'X values in the input files')
    p.add_argument('--y-col', type=int, default=1,
                  help='column number (index starting from 0) of '
                  'Y values in the input files')
    p.add_argument('fn', metavar='fn', type=str, nargs='+',
                   help='filenames containing different results')
    p.add_argument('--savefig', type=str, default='fig.png',
                   help='save figure in file')
    return p.parse_args()

cfg = parse_args()

def main():
    colors = []
    save_fn = cfg.savefig
    pl.rc('axes', color_cycle=['r', 'g', 'b', 'y', 'c', 'm', 'k'])
    for fn in cfg.fn:
        x = []
        y = []
        label = []
        with open(fn) as f:
            for l in f:
                fd = l.split('\t')
                x.append(float(fd[cfg.x_col]))
                y.append(float(fd[cfg.y_col]))
        label = fn.split('/')[-1].split('.')[0]
        pl.plot(x, y, label=label)
    pl.xlabel(cfg.xlabel)
    pl.ylabel(cfg.ylabel)
    pl.title('{0} - {1}'.format(cfg.xlabel, cfg.ylabel))
    pl.legend(loc=cfg.legend_pos, shadow=True, fontsize='small')
    pl.savefig(save_fn)


if __name__ == '__main__':
    main()
