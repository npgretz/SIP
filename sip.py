#!/usr/bin/env python2

# sip.py - generate summary measures from accelerometer data
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

import sys
import subprocess as sp
import pathlib
import argparse
import textwrap

import numpy as np
import pandas as pd

from pandas.tseries.offsets import Second

import sip_utils as util

class ActivitySummary(util.Sniffable):
    def __init__(self, data):
        self.data = data

    def to_file(self, filepath):
        self.data.to_csv(str(filepath), index=False)

    @classmethod
    def from_sojourns(cls, sojourns, subj):
        # FIXME this whole method is horrifying
        durs = range(10, 61, 10)
        day_idx = util.days(sojourns.raw_data.index)
        # FIXME does this still work if we pass in the index?
        out = cls.prepare_output().reindex(index=day_idx[:-1])
        out_cols = set(out.columns)

        out['subject'] = subj
        out['day'] = util.WEEKDAYS[out.index.weekday]
        out['sleep_ranges'] = map(sleep_ranges,
                                  util.slice_data(sojourns.data, day_idx))
        out['total_counts'] = [total(day, 'counts') for day in
                                   util.slice_data(sojourns.raw_data, day_idx)]
        if 'steps' in sojourns.raw_data.columns:
            out['AG_steps'] = [total(day, 'steps') for day in
                                   util.slice_data(sojourns.raw_data, day_idx)]
        if 'AP.steps' in sojourns.raw_data.columns:
            out['AP_steps'] = [2*total(day, 'AP.steps') for day in
                                   util.slice_data(sojourns.raw_data, day_idx)]
        for classifier in util.classifiers:
            grouped = util.group_by_classifier(sojourns.data, classifier)
            sliced = util.slice_data(grouped, day_idx)
            hours = util.clock_hours(grouped.index)
            key = 'min_%s' % classifier.name
            if key in out_cols:
                out[key] = [delta(day)[day[classifier.cname].notnull()].sum() /
                                pd.Timedelta(1, 'm')
                            for day in sliced]
            key = '%s_periods' % classifier.name
            if key in out_cols:
                out[key] = [day[classifier.cname].nunique() for day in sliced]
            key = 'mean_%s_len' % classifier.name
            # Weight by minute, not by bout.
            if key in out_cols:
                out[key] = [(delta(day) / delta(day).where(
                                     day[classifier.cname].notnull()).sum() *
                                 day['bout_Dur']).sum() / pd.Timedelta(1, 'm')
                            for day in sliced]
            # Include bouts that extend into neighboring days
            key = 'median_%s_len' % classifier.name
            if key in out_cols:
                out[key] = [day.groupby(classifier.cname).first()['bout_Dur']
                                .median() / pd.Timedelta(1, 'm')
                            for day in sliced]
            # Incorrectly allow intensity to bleed across midnight because the
            # current architecture doesn't allow us to do this right.
            key = 'mean_%s_intensity' % classifier.name
            if key in out_cols:
                out[key] = [delta(day) / delta(day).where(
                                    day[classifier.cname].notnull()).sum() *
                                day['counts'].sum()
                            for day in sliced]
            # Unused
            # This is wrong; it can't be computed from processed data
#            key = 'median_%s_intensity' % classifier.name
#            if key in out_cols:
#                out[key] = [day.ix[day[classifier.cname].notnull(),'counts']
#                                .median()
#                            for day in sliced]
            for dur in durs:
                key = '%s_length_%d' % (classifier.name, dur)
                if key in out_cols:
                    out[key] = [delta(day)[day[classifier.cname].notnull() &
                                               (day['bout_Dur'] >=
                                                    pd.Timedelta(dur, 'm'))] \
                                    .sum() / pd.Timedelta(1, 'm')
                                for day in sliced]
            for hh, hour in zip(hours, util.slice_data(grouped, hours)):
                # unlike sliced, hour can be empty
                key = '%s_circadian_%d' % (classifier.name, hh.hour)
                if key in out_cols:
                    out.ix[pd.Timestamp(hh.date(), tz=util.tz),key] = \
                        delta(hour)[hour[classifier.cname].notnull()].sum() / \
                            pd.Timedelta(1, 'm')
                for dur in durs:
                    key = '%s_circadian_%d_length_%d' % (classifier.name,
                                                         hh.hour, dur)
                    if key in out_cols:
                        out.ix[pd.Timestamp(hh.date(), tz=util.tz),key] = \
                            delta(hour)[hour[classifier.cname].notnull() &
                                           (hour['bout_Dur'] >=
                                                pd.Timedelta(dur, 'm'))
                                       ].sum() / pd.Timedelta(1, 'm')
            out['counts_per_min'] = out['total_counts'] / out['min_awake']
            out['break_rate'] = (60*out['sedentary_periods'] /
                out['min_sedentary'])

        return cls(out)

    @staticmethod
    def prepare_output():
        durs = range(10, 61, 10)
        hours = range(24)
        dtypes = [('subject', 'O'),
                  ('day', 'O'),
                  ('min_awake', 'f8'),
                  ('sleep_ranges', 'O'),
                  ('total_counts', 'i8'),
                  ('counts_per_min', 'f8'),
                  ('min_sedentary', 'f8'),
                  ('sedentary_periods', 'i8'),
                  ('mean_sedentary_len', 'f8'),
                  ('median_sedentary_len', 'f8'),
                  ('active_periods', 'i8'),
                  ('mean_active_len', 'f8'),
                  ('median_active_len', 'f8')]
        dtypes.extend([('min_%s' % act, 'f8') for act in
                       'active light adl freedson matthews vigorous'.split()])
        dtypes.extend([('median_active_intensity', 'i8'),
                       ('min_pa_recommendation', 'f8'),
                       ('pa_recommendation_periods', 'i8')])
        dtypes.extend([('%s_length_%d' % (act, dur), 'f8') for act in
                       'sedentary light adl freedson matthews vigorous'.split()
                       for dur in durs])
        dtypes.extend([('%s_circadian_%d' % (act, hour), 'f8') for act in
                       'sedentary light adl freedson matthews vigorous'
                       '    awake'.split()
                       for hour in hours])
        dtypes.extend([('%s_circadian_%d_length_%d' % (act, hour, dur), 'f8')
                       for act in 'sedentary'.split()
                       for dur in durs for hour in hours])
        dtypes.append(('break_rate', 'f8'))
        dtypes.extend([('min_%s' % act, 'f8') for act in
                       'standing sitting_active'.split()])
        dtypes.extend([('%s_length_%d' % (act, dur), 'f8') for act in
                       'standing sitting_active'.split() for dur in durs])
        dtypes.extend([('%s_circadian_%d' % (act, hour), 'f8') for act in
                       'standing sitting_active'.split() for hour in hours])
        dtypes.append(('min_valid', 'f8'))
        dtypes.extend([('valid_circadian_%d' % hour, 'f8') for hour in hours])
        dtypes.extend([('min_%s' % act, 'f8') for act in
                       'adl_apsed freedson_apsed vigorous_apsed'.split()])
        dtypes.append(('min_error', 'f8'))
        dtypes.extend([('min_%s_and_pa_recommendation' % act, 'f8') for act in
                       'adl freedson vigorous'.split()])
        dtypes.extend([('%s_steps' % meth, 'i8') for meth in ('AG', 'AP')])
        return pd.DataFrame(np.zeros(0, dtype=dtypes))

    @staticmethod
    def search_path():
        return ['ActiGraph/*_processed.csv',
                '*_processed.csv']

def create_sojourns(ag_path, ap_path=None, soj_path=None):
    # TODO: move this onto SojournsData as a classmethod
    """
    Process an ActiGraph dataset using Sojourns.

    If an optional activPAL dataset is specified, use SIP instead to improve
    the handling of sedentary vs. standing time.

    """
    # PERF: Currently, this is the #1 time sink.
    util.patch_subprocess()
    if not soj_path:
        soj_path = ag_path.with_name(ag_path.stem +
                                     ('_with_activpal' if ap_path else '') +
                                     '_sojourns.csv')
    # This relies on the fact that python and R use sufficiently similar
    # string literal syntaxes that eval-ing in R the __repr__ of a python
    # string is safe and correct.  This is currently true for python2 but not
    # python3.
    # FIXME: %r was cool until it meant explicitly calling str() on the paths
    sip_dir = pathlib.Path(__file__).parent
    r_cmds = ("""
              load(paste0(%r, "/nnet3ests.RData"))
              load(paste0(%r, "/cent.1.RData"))
              load(paste0(%r, "/scal.1.RData"))
              load(paste0(%r, "/class.nnn.use.this.RData"))
              source(paste0(%r, "/sip.functions.R"))
              library(nnet)
              data <- AG.file.reader(%r)
              """ % ((str(sip_dir),)*5 + (str(ag_path),)) +
             ("""
              ap <- AP.file.reader(%r)
              data <- enhance.actigraph(data,ap)
              """ % str(ap_path) if ap_path else "") +
              """
              sip.estimate <- sojourn.3x(data)
              sojourns.file.writer(sip.estimate,%r)
              """ % str(soj_path))
    with sp.Popen(['R', '--vanilla'], stdin=sp.PIPE) as p:
        p.communicate(r_cmds)
        if p.returncode:
            raise sp.CalledProcessError(returncode=p.returncode, cmd='R')

def sleep_ranges(day):
    # FIXME: if DatetimeIndex gets vectorized strftime, use it
    day = util.contiguous_apply(
        day,
        day['awake'],
        lambda v, blk: pd.DataFrame({'t': [blk.index[0].strftime(util.dtfmt)],
                                     'awake': [v]}, index=blk.index[[0]]))
    return '[%s]' % '; '.join(
        map('...'.join,
            zip(day.ix[~day['awake'].astype(bool),'t'],
                day.ix[~day['awake'].shift().astype(bool),'t'])))

def total(rawday, col):
    return rawday.ix[rawday['awake'].astype(bool),col].sum()

def delta(data):
    out = pd.Series(index=data.index)
    out[:-1] = np.diff(data.index.values)
    return out

def parse_arguments():
    usage = '%(prog)s [options] [path/to/subject/directory]'
    description = textwrap.dedent("""
    Process accelerometer data using Sojourns/SIP.

    Proceed in two steps: first, use input accelerometer data to estimate
    wear time and metabolic activity; second, generate many summary
    statistics from these estimates.

    There are two major ways to get input data to this program: you can give
    it the path to your subject's data and have it work out which files are
    where, or you can tell it precisely where to find each input file. You
    can also use the first method and then override specific defaults, if
    you prefer.

    The defaults have been chosen so that you can download your activity
    monitor data directly from the device to a subject's directory and run
    this program specifying only that directory.

    By default, this program searches for files named as in the examples
    below, except with less Victor Hugo. It will also find files directly in
    the subject directory, as well as files with names ending in "_QC.csv",
    which it will use preferentially to allow quality-controlling data
    without editing the original files.

    Don't store data from more than one subject in the same directory; if
    you do, this program will get confused and may mix subjects' data by
    accident!

    Input files:

    - ActiGraph data in 1-second epochs, as generated by the ActiLife
      software. This file must exist in order to complete the first step.

      Example file name: 24601/ActiGraph/JV_1secDataTable.csv

      Set this with `--ag-path FILENAME`.

    - activPAL data, as generated by the activPAL software. These consist of
      two files. If these files are found, use the SIP method in the first
      step; otherwise, use the original Sojourns method.

      Example file name: "24601/activPAL/JV Events.csv"
      Must also exist:   24601/activPAL/JV.def

      (You must quote this file name on the command line because it contains
      a space.)

      Note that these file names must have the *exact* same stem (here
      "JV"). The filenames generated by the activPAL software do this by
      default.

      Set this with `--ap-path FILENAME`.

    Intermediate files:

    - Awake ranges data, indicating when the subject was wearing the
      monitor(s). This file is generated by this program, but if a modified
      version already exists it will be used instead of estimating this
      information. This allows you to account for instances when the subject
      fell asleep while wearing the monitor, for instance.

      Example file name: "24601/ActiGraph/JV awake ranges.csv"

      (You must quote this file name on the command line because it contains
      a space.)

      You can edit this file in Excel, but if you do, you must take care to
      always delete cells rather than clearing their contents. Also, make
      sure to save as a CSV file.

      Set this with `--awake-path "FILENAME"`, or ignore an
      existing awake ranges file with `--ignore-awake-ranges`.

    - Sojourns/SIP annotated data, indicating bout boundaries and second-by-
      second estimated metabolic activity. This file is generated by this
      program, but if it already exists it will not be recomputed to save
      time. Editing this file by hand is not recommended.

      Example file name:
      24601/ActiGraph/JV_1secDataTable_with_activpal_sojourns.csv

      By default, this path will be the same as the ActiGraph data with
      "_sojourns" or "_with_activpal_sojourns" added before the ".csv",
      depending on whether activPAL data have been provided.

      Set this with `--soj-path FILENAME`.

    Output files:

    - Sojourns/SIP processed data, containing loads of summary measures
      generated from the metabolic estimates. See the README for a detailed
      description of the contents of this file.

      Example file name:
      24601/ActiGraph/JV_1secDataTable_with_activpal_sojourns_processed.csv

      This will always use the Sojourns/SIP file path with "_processed"
      added before the ".csv".

    Because many of the summary measures refer to times of day, it's
    important to provide the time zone in which the data were collected if
    it's different from the system time zone of the computer doing the
    processing. (Use the IANA time zone, like "America/Chicago", not the
    ambiguous abbreviation like "CST", which could mean Cuba Standard Time.)

    """)    # TODO: summary measures
    parser = argparse.ArgumentParser(
        usage=usage, description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('subjdir', type=pathlib.Path, nargs='?',
                        help='search for subject data in this directory')
    parser.add_argument('-s', '--subject', dest='subj',
                        help='embed this tag as the subject identifier into '
                             'the processed output; the default value is the '
                             'name of the subject directory ("24601" in the '
                             'examples)')
    parser.add_argument('--ag-path', type=pathlib.Path,
                        help='get ActiGraph 1secDataTable data from this file')
    parser.add_argument('--ap-path', type=pathlib.Path,
                        help='get activPAL Events data from this file')
    parser.add_argument('--soj-path', type=pathlib.Path,
                        help='write Sojourns/SIP estimated metabolic activity '
                             "to this file if it doesn't already exist; "
                             'otherwise, read previously computed metabolic '
                             'estimates from this file (to save time)')
    parser.add_argument('--awake-path', type=pathlib.Path,
                        help='read wear time intervals from this file if it '
                             'exists; otherwise, estimate wear time and write '
                             'the estimates to this file')
    parser.add_argument('--soj-intermediate-path', type=pathlib.Path,
                        help=argparse.SUPPRESS)
    parser.add_argument('--ignore-awake-ranges', action='store_true',
                        help='ignore an existing "awake ranges" file and '
                             'estimate wear time anyway')
    parser.add_argument('--tz',
                        help='interpret data as being collected in this time '
                             'zone instead of %r' %
                                 getattr(util.tz, 'zone', util.tz))
    args = parser.parse_args()
    if args.tz is not None:
        util.tz = args.tz
    if args.subjdir is not None:
        if not args.subj:
            args.subj = args.subjdir.resolve().parts[-1]
        if not args.ag_path:
            args.ag_path = util.ActiGraphDataTable.sniff(args.subjdir,
                                                         epoch=Second())
        if not args.ap_path:
            args.ap_path = util.ActivPALData.sniff(args.subjdir)
        if not args.soj_path:
            args.soj_path = util.SojournsData.sniff(args.subjdir)
        if not args.awake_path:
            args.awake_path = util.AwakeRanges.sniff(args.subjdir)
    if not args.ag_path and not args.soj_path:
        if args.subjdir is not None:
            if not args.subjdir.exists():
                raise IOError("can't find subject directory %r" %
                              str(args.subjdir))
            elif not args.subjdir.is_dir():
                raise IOError("subjdir %r isn't a directory" %
                              str(args.subjdir))
            raise IOError("can't find any data in subject directory %r" %
                          str(args.subjdir))
        parser.print_help()
        parser.exit()
    return args

if __name__ == '__main__':
    # FIXME mess
    import sys

    args = parse_arguments()
    subj = args.subj
    ag_path = args.ag_path
    ap_path = args.ap_path
    soj_path = args.soj_path
    awake_path = args.awake_path
    ignore_awake_ranges = args.ignore_awake_ranges

    if not soj_path or (ag_path and ap_path and
                        'activpal' not in soj_path.name):
        if not ag_path:
            raise OSError("Can't find ActiGraph input to sojourns")
        soj_path = ag_path.with_name(ag_path.stem +
                                     ('_with_activpal' if ap_path else '') +
                                     '_sojourns.csv')
    if not awake_path:
        if ag_path and ag_path.name.endswith('1secDataTable.csv'):
            awake_path = ag_path.with_name(
                ag_path.name.replace('.csv', ' awake ranges.csv'))
        elif ap_path and ap_path.name.endswith('Events.csv'):
            awake_path = ap_path.with_name(
                ap_path.name.replace('Events.csv', 'awake ranges.csv'))
    print('Subject: %s' % subj)
    print('soj_path: %s' % soj_path)
    print('ag_path: %s' % ag_path)
    print('ap_path: %s' % ap_path)
    print('awake_path: %s' % awake_path)
    sys.stdout.flush()
    if not soj_path.exists():
        create_sojourns(ag_path, ap_path, soj_path)
    if awake_path and not ignore_awake_ranges and awake_path.exists():
        awake_ranges = util.AwakeRanges.from_file(awake_path)
    else:
        awake_ranges = None
    soj = util.SojournsData.from_file(soj_path, awake_ranges)
    # XXX HACK
    if 'steps' not in soj.raw_data.columns:
        ag = util.ActiGraphDataTable.from_file(ag_path, awake_ranges)
        soj.raw_data['steps'] = ag.raw_data['Steps']
    soj.process()
    # FIXME integrate this better
    if args.soj_intermediate_path:
        soj.data.iloc[:-1].to_csv(str(args.soj_intermediate_path))
        util.contiguous_apply(
            soj.data, soj.data['awake'], lambda val, block: pd.DataFrame(
                {'counts': block['counts'].sum(),
                 'Dur': block['Dur'].sum(),
                 'METs': (block['Dur'] / block['Dur'].sum() *
                     block['METs']).sum(),
                 'ActivityLevel': ','.join(block['ActivityLevel'].dropna()
                     .unique())},
                index=[block.index[0]]) if val and not np.isnan(val) else
            pd.DataFrame(
                {'counts': [], 'Dur': [], 'METs': [], 'ActivityLevel':[]}
            )).to_csv(
            str(args.soj_intermediate_path.with_name(
                    args.soj_intermediate_path.name.replace(
                        args.soj_intermediate_path.suffix,
                        '_squashed'+args.soj_intermediate_path.suffix))))
    if awake_path and not awake_path.exists():
        soj.awake_ranges.to_file(awake_path)
    summary = ActivitySummary.from_sojourns(soj, subj)
    outpath = soj_path.with_name(soj_path.stem + '_processed.csv')
    summary.to_file(outpath)
