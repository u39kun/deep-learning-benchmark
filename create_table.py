import argparse
import pickle
from tabulate import tabulate

from benchmark import models, precisions, frameworks

def print_as_table(file):
    results = pickle.load(open(file, 'rb'))

    rows = []
    for fw in frameworks:
        for precision in precisions:
            precision_display = '32-bit' if precision == 'fp32' else '16-bit'
            if fw in results:
                row = ['{:.1f}ms'.format(v) if v > 0 else '' for v in results[fw][precision]]
                rows.append([fw, precision_display] + row)

    header = ['{} {}'.format(m, phase) for m in models for phase in ['eval', 'train']]
    header = ['Framework', 'Precision'] + header
    table = tabulate(rows, header, tablefmt="pipe")
    print(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='file', required=True)
    args = parser.parse_args()
    print_as_table(args.file)
