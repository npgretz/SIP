#!/usr/bin/env python2

from __future__ import print_function

import itertools as it
import operator as op
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas.tseries.offsets import Minute, Second

from sip_utils import *

def prepare_axis(dates):
    fig, ax = plt.subplots()
    daylen = pd.offsets.Day().nanos
    hourlen = pd.offsets.Hour().nanos
    xmin = (daylen - np.diff(dates.values)).min().astype('i8')
    ax.set_xlim([xmin, daylen])
    ax.set_xticks(np.arange(xmin, daylen+1, hourlen))
    ax.set_xticklabels(ax.get_xticks()/hourlen)
    ax.set_ylim([-0.5, len(dates)-1.5])
    ax.set_yticks(range(len(dates)-1))
    ax.set_yticklabels(dates[:-1].map(
        op.methodcaller('strftime', '%a %Y-%m-%d')))
    ax.invert_yaxis()
    plt.tight_layout()
    return ax

def plot_all_series(data, eps=0.1):
    dates = reduce(pd.DatetimeIndex.union, [days(d.data.index) for d in data])
    ax = prepare_axis(dates)
    n = len(data)
    intervals = np.column_stack([np.linspace(eps-0.5, 0.5-eps, n, False),
                                 [(1.0-2*eps)/n]*n])
    for interval, d in zip(intervals, data):
        d.plot(dates[0], ax, interval)
    plt.show()

def parse_options():
    from optparse import OptionParser, OptionValueError, SUPPRESS_HELP
    from textwrap import dedent
    patch_optparse()

    usage = 'Usage: %prog [options] [path/to/subject/directory]'
    description = dedent("""
    Display activity monitor data in an attractive format.

    If a path to the subject's directory is given as the final argument,
    %prog will search there for files to display, so that the --*-path
    options will not be necessary.

    """)
    epilog = dedent("""
    You may specify the --soj-path, --ag-path, --ap-path and --exclude
    options as many times as you like; each file specified this way will be
    plotted or ignored as directed.

    """)
    parser = OptionParser(description=description, epilog=epilog)
    parser.add_option('--soj-path', type='path', action='append', default=[],
                      help='Path to sojourns preprocessed Actigraph data')
    parser.add_option('--ag-path', type='path', action='append', default=[],
                      help='Path to Actigraph data')
    parser.add_option('--ap-path', type='path', action='append', default=[],
                      help='Path to activPAL Events data')
    parser.add_option('--awake-path', type='path',
                      help='Path to "awake ranges" file in case autodetection '
                           'of non-wear time is poor')
    parser.add_option('--ignore-awake-ranges', action='store_true',
                      help='Flag to ignore "awake ranges" file')
    parser.add_option('--no-raw-counts', action='store_true',
                      help="Don't plot raw counts (for speed reasons)")
    parser.add_option('-x', '--exclude', type='path', action='callback',
                      nargs=0, default=set(), callback=exclude_callback,
                      help="Don't plot the data in this file")
    (opts, args) = parser.parse_args()
    if args:
        try:
            path, = map(Path, args)
        except ValueError:
            path = None
        else:
            if not opts.ag_path:
                opts.ag_path = filter(
                    None, [ActiGraphDataTable.sniff(path, epoch=Minute()),
                           ActiGraphDataTable.sniff(path, epoch=Second())])
            if not opts.ap_path:
                opts.ap_path = filter(None, [ActivPALData.sniff(path)])
            if not opts.soj_path:
                opts.soj_path = filter(None, [SojournsData.sniff(path)])
            if not opts.awake_path:
                opts.awake_path = AwakeRanges.sniff(path)
    return opts

if __name__ == '__main__':
    opts = parse_options()
    # FIXME
    if opts.no_raw_counts:
        SojournsData.plot = SojournsData.plot_activities

    data = []
    if opts.awake_path and opts.awake_path.exists():
        awake_ranges = AwakeRanges.from_file(opts.awake_path)
    else:
        awake_ranges = None
    # FIXME de-hackify
    print('ag_paths: [\n   ', '\n    '.join(map(str, opts.ag_path)), '\n]\n')
    print('soj_paths: [\n   ', '\n    '.join(map(str, opts.soj_path)), '\n]\n')
    print('ap_paths: [\n   ', '\n    '.join(map(str, opts.ap_path)), '\n]\n')
    print('exclude: {\n   ', '\n    '.join(map(str, opts.exclude)), '\n}\n')
    print('awake_path: %s' % opts.awake_path)
    for ag_path in opts.ag_path:
        if ag_path in opts.exclude:
            continue
        # FIXME hack
        ag = (ActiGraphUpdatedDataTable
              if 'updated' in ag_path.name
              else ActiGraphDataTable).from_file(ag_path,
                                                 awake_ranges=awake_ranges)
        if ag.raw_data.index.freq == Minute():
            ag.process()
        else:
            ag.raw_data['awake'] = ag.find_sleep()
            ag.data = ag.raw_data
        data.append(ag)
    for soj_path in opts.soj_path:
        if soj_path in opts.exclude:
            continue
        soj = SojournsData.from_file(soj_path, awake_ranges=awake_ranges)
        soj.process()
        data.append(soj)
    for ap_path in opts.ap_path:
        ap = ActivPALData.from_file(ap_path)
        data.append(ap)

    plot_all_series(data)
