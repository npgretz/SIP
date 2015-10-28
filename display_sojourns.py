#!/usr/bin/env python2

# display_sojourns.py - display accelerometer data in an attractive format
# Copyright (C) 2015 Isaac Schwabacher
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import itertools as it
import operator as op
import argparse
import pathlib
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas.tseries.offsets import Minute, Second

import sip_utils as util

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
    dates = reduce(pd.DatetimeIndex.union,
                   [util.days(d.data.index) for d in data])
    ax = prepare_axis(dates)
    n = len(data)
    intervals = np.column_stack([np.linspace(eps-0.5, 0.5-eps, n, False),
                                 [(1.0-2*eps)/n]*n])
    for interval, d in zip(intervals, data):
        d.plot(dates[0], ax, interval)
    plt.show()

def parse_arguments():
    usage = '%(prog)s [options] [path/to/subject/directory]'
    description = textwrap.dedent("""
    Display activity monitor data in an attractive format.

    Display each type of data differently:

    - ActiGraph data integrated to 1-second epochs are displayed as a line
      plot of the counts along each axis.

      The blue line represents the first (vertical) axis, the green line the
      second (anterior-posterior) axis, and the red line the third (medial-
      lateral) axis.

    - ActiGraph data integrated to 60-second epochs are classified according
      to the modified (Freedson 1998) cut points used by the ActiLife
      software and displayed as bars color coded by estimated intensity.

      The colors correspond to estimated intensities as follows:
      * Dark blue:    non-wear
      * Blue:         sedentary
      * Light yellow: light
      * Yellow:       lifestyle
      * Orange:       moderate
      * Red:          vigorous

    - Sojourns/SIP data are displayed as bars color coded by estimated
      intensity.

      The colors are as above, with additional colors as follows:
      * Green:        standing
      * Cyan:         seated, but light (in practice, this tends to indicate
                      activities like recumbent biking, the intensities of
                      which are typically underestimated)
      * Black:        Sojourns estimated negative intensity for this bout
                      (this is an inherent problem with the method but can
                      only happen when Sojourns has already classified a
                      bout as active; such bouts are typically moderate or
                      vigorous)

    - activPAL Events data are displayed as bars color coded by whether the
      subject was sitting, standing or stepping.

      Here sitting is blue, standing green, and stepping red.

    If the graph crosses a time change (for instance, as caused by Daylight
    Saving Time), data which occurs before the change but on the same day
    will be shifted to fit.

    Files are selected in the same way as in sip.py; for more detail, see
    the help for that playing. The exception to this is that this program
    will select as many files as it can find rather than ending its search
    when it finds an appropriate file (but files with names ending in "_QC"
    will still shadow files with identical names that are missing this
    suffix).

    If you wish to exclude a particular file from being plotted, you can
    pass it to the `--exclude` option.

    """)
    epilog = textwrap.dedent("""
    You may specify the --soj-path, --ag-path, --ap-path and --exclude
    options as many times as you like; each file specified this way will be
    plotted or ignored as directed.

    """)

    parser = argparse.ArgumentParser(
        usage=usage, description=description, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subjdir', type=pathlib.Path, nargs='?',
                        help='search for subject data in this directory')
    parser.add_argument('--soj-path', type=pathlib.Path, action='append',
                        help='get Sojourns/SIP preprocessed Actigraph data '
                             'from this file')
    parser.add_argument('--ag-path', type=pathlib.Path, action='append',
                        help='get Actigraph data from this file')
    parser.add_argument('--ap-path', type=pathlib.Path, action='append',
                        help='get activPAL events data from this file')
    parser.add_argument('--awake-path', type=pathlib.Path,
                        help='get wear time intervals from this file in case '
                             'autodetection of non-wear time is poor')
    parser.add_argument('--ignore-awake-ranges', action='store_true',
                        help='ignore "awake ranges" file')
    parser.add_argument('--no-raw-counts', action='store_true',
                        help="don't plot raw counts (for speed reasons)")
    parser.add_argument('-x', '--exclude', type=pathlib.Path, action='append',
                        help="don't plot the data in this file")
    parser.add_argument('--tz', default=util.tz,
                        help='interpret data as being collected in this time '
                             'zone instead of %(default)r')
    args = parser.parse_args()
    util.tz = args.tz
    if args.subjdir is not None:
        if not args.ag_path:
            args.ag_path = filter(
                None, [util.ActiGraphDataTable.sniff(args.subjdir,
                                                     epoch=Minute()),
                       util.ActiGraphDataTable.sniff(args.subjdir,
                                                     epoch=Second())])
        if not args.ap_path:
            args.ap_path = filter(
                None, [util.ActivPALData.sniff(args.subjdir)])
        if not args.soj_path:
            args.soj_path = filter(
                None, [util.SojournsData.sniff(args.subjdir)])
        if not args.awake_path:
            args.awake_path = util.AwakeRanges.sniff(args.subjdir)
    if args.ignore_awake_ranges:
        args.awake_path = None
    return args

if __name__ == '__main__':
    args = parse_arguments()
    args.exclude = set(args.exclude if args.exclude is not None else [])
    # FIXME
    if args.no_raw_counts:
        util.SojournsData.plot = util.SojournsData.plot_activities

    data = []
    if args.awake_path and args.awake_path.exists():
        awake_ranges = util.AwakeRanges.from_file(args.awake_path)
    else:
        awake_ranges = None
    # FIXME de-hackify
    print('ag_paths: [\n   ', '\n    '.join(map(str, args.ag_path)), '\n]\n')
    print('soj_paths: [\n   ', '\n    '.join(map(str, args.soj_path)), '\n]\n')
    print('ap_paths: [\n   ', '\n    '.join(map(str, args.ap_path)), '\n]\n')
    print('exclude: {\n   ', '\n    '.join(map(str, args.exclude)), '\n}\n')
    print('awake_path: %s' % args.awake_path)
    for ag_path in args.ag_path:
        if ag_path in args.exclude:
            continue
        ag = util.ActiGraphDataTable.from_file(ag_path,
                                               awake_ranges=awake_ranges)
        if ag.raw_data.index.freq == Minute():
            ag.process()
        else:
            ag.raw_data['awake'] = ag.find_sleep()
            ag.data = ag.raw_data
        data.append(ag)
    for soj_path in args.soj_path:
        if soj_path in args.exclude:
            continue
        soj = util.SojournsData.from_file(soj_path, awake_ranges=awake_ranges)
        soj.process()
        data.append(soj)
    for ap_path in args.ap_path:
        ap = util.ActivPALData.from_file(ap_path)
        data.append(ap)

    plot_all_series(data)
