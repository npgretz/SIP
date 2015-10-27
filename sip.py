#!/usr/bin/env python2

from __future__ import print_function

import subprocess as sp
from pathlib import Path

import numpy as np
import pandas as pd

from pandas.tseries.offsets import Second

from sip_utils import *

class ActivitySummary(Sniffable):
    def __init__(self, data):
        self.data = data

    def to_file(self, filepath):
        self.data.to_csv(str(filepath), index=False)

    @classmethod
    def from_sojourns(cls, sojourns, subj):
        # FIXME this whole method is horrifying
        durs = range(10, 61, 10)
        day_idx = days(sojourns.raw_data.index)
        # FIXME does this still work if we pass in the index?
        out = cls.prepare_output().reindex(index=day_idx[:-1])
        out_cols = set(out.columns)

        out['subject'] = subj
        out['day'] = WEEKDAYS[out.index.weekday]
        out['sleep_ranges'] = map(sleep_ranges,
                                  slice_data(sojourns.data, day_idx))
        out['total_counts'] = [total(day, 'counts') for day in
                                  slice_data(sojourns.raw_data, day_idx)]
        if 'steps' in sojourns.raw_data.columns:
            out['AG_steps'] = [total(day, 'steps') for day in
                                   slice_data(sojourns.raw_data, day_idx)]
        if 'AP.steps' in sojourns.raw_data.columns:
            out['AP_steps'] = [2*total(day, 'AP.steps') for day in
                                   slice_data(sojourns.raw_data, day_idx)]
        for classifier in classifiers:
            grouped = group_by_classifier(sojourns.data, classifier)
            sliced = slice_data(grouped, day_idx)
            hours = clock_hours(grouped.index)
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
            for hh, hour in zip(hours, slice_data(grouped, hours)):
                # unlike sliced, hour can be empty
                key = '%s_circadian_%d' % (classifier.name, hh.hour)
                if key in out_cols:
                    out.ix[pd.Timestamp(hh.date(), tz=tz),key] = \
                        delta(hour)[hour[classifier.cname].notnull()].sum() / \
                            pd.Timedelta(1, 'm')
                for dur in durs:
                    key = '%s_circadian_%d_length_%d' % (classifier.name,
                                                         hh.hour, dur)
                    if key in out_cols:
                        out.ix[pd.Timestamp(hh.date(), tz=tz),key] = \
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
                  ('min_awake', 'f64'),
                  ('sleep_ranges', 'O'),
                  ('total_counts', 'i64'),
                  ('counts_per_min', 'f64'),
                  ('min_sedentary', 'f64'),
                  ('sedentary_periods', 'i64'),
                  ('mean_sedentary_len', 'f64'),
                  ('median_sedentary_len', 'f64'),
                  ('active_periods', 'i64'),
                  ('mean_active_len', 'f64'),
                  ('median_active_len', 'f64')]
        dtypes.extend([('min_%s' % act, 'f64') for act in
                       'active light adl freedson matthews vigorous'.split()])
        dtypes.extend([('median_active_intensity', 'i64'),
                       ('min_pa_recommendation', 'f64'),
                       ('pa_recommendation_periods', 'i64')])
        dtypes.extend([('%s_length_%d' % (act, dur), 'f64') for act in
                       'sedentary light adl freedson matthews vigorous'.split()
                       for dur in durs])
        dtypes.extend([('%s_circadian_%d' % (act, hour), 'f64') for act in
                       'sedentary light adl freedson matthews vigorous'
                       '    awake'.split()
                       for hour in hours])
        dtypes.extend([('%s_circadian_%d_length_%d' % (act, hour, dur), 'f64')
                       for act in 'sedentary'.split()
                       for dur in durs for hour in hours])
        dtypes.append(('break_rate', 'f64'))
        dtypes.extend([('min_%s' % act, 'f64') for act in
                       'standing sitting_active'.split()])
        dtypes.extend([('%s_length_%d' % (act, dur), 'f64') for act in
                       'standing sitting_active'.split() for dur in durs])
        dtypes.extend([('%s_circadian_%d' % (act, hour), 'f64') for act in
                       'standing sitting_active'.split() for hour in hours])
        dtypes.append(('min_valid', 'f64'))
        dtypes.extend([('valid_circadian_%d' % hour, 'f64') for hour in hours])
        dtypes.extend([('min_%s' % act, 'f64') for act in
                       'adl_apsed freedson_apsed vigorous_apsed'.split()])
        dtypes.append(('min_error', 'f64'))
        dtypes.extend([('min_%s_and_pa_recommendation' % act, 'f64') for act in
                       'adl freedson vigorous'.split()])
        dtypes.extend([('%s_steps' % meth, 'i64') for meth in ('AG', 'AP')])
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
    patch_subprocess()
    if not soj_path:
        soj_path = ag_path.with_name(ag_path.stem +
                                     ('_with_activpal' if ap_path else '') +
                                     '_sojourns.csv')
    # This relies on the fact that python and R use sufficiently similar
    # string literal syntaxes that eval-ing in R the __repr__ of a python
    # string is safe and correct.  This is currently true for python2 but not
    # python3.
    # FIXME: %r was cool until it meant explicitly calling str() on the paths
    r_cmds = ("""
              load(paste0(%r, "/nnet3ests.RData"))
              load(paste0(%r, "/cent.1.RData"))
              load(paste0(%r, "/scal.1.RData"))
              load(paste0(%r, "/class.nnn.use.this.RData"))
              source(paste0(%r, "/sip.functions.R"))
              library(nnet)
              data <- AG.file.reader(%r)
              """ % ((str(Path(__file__).parent),)*5 + (str(ag_path),)) +
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
    day = contiguous_apply(day, day['awake'], lambda v, blk:
                           pd.DataFrame({'t': [blk.index[0].strftime(dtfmt)],
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

def parse_options():
    from optparse import OptionParser, SUPPRESS_HELP
    from textwrap import dedent
    patch_optparse()

    usage = 'Usage: %prog [options] [path/to/subject/directory]'
    description = dedent("""
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
    below, except with less Sean Connery. It will also find files directly
    in the subject directory, as well as files with names ending in
    "_QC.csv", which it will use preferentially to allow quality-controlling
    data without editing the original files.

    Don't store data from more than one subject in the same directory; if
    you do, this program will get confused and may mix subjects' data by
    accident!

    Input files:

    - ActiGraph data in 1-second epochs, as generated by the ActiLife
      software. This file must exist in order to complete the first step.

      Example file name: "007/ActiGraph/James_1secDataTable.csv"

      Set this with `--ag-path FILENAME`.

    - activPAL data, as generated by the activPAL software. These consist of
      two files. If these files are found, use the SIP method in the first
      step; otherwise, use the original Sojourns method.

      Example file names: "007/activPAL/James Events.csv" and
                          "007/activPAL/James.def"

      Note that these file names must have the *exact* same stem (here
      "James"). The filenames generated by the activPAL software do this by
      default.

      Set this with `--ap-path FILENAME`, making sure to name the Events.csv
      file rather than the .def file.

    Intermediate files:

    - Awake ranges data, indicating when the subject was wearing the
      monitor(s). This file is generated by this program, but if a modified
      version already exists it will be used instead of estimating this
      information. This allows you to account for instances when the subject
      fell asleep while wearing the monitor, for instance.

      Example file name: "007/ActiGraph/James_awake ranges.csv"

      You can edit this file in Excel, but if you do, you must take care to
      always delete cells rather than clearing their contents. Also, make
      sure to save as a CSV file.

      Set this with `--awake-path "your_path_here.csv"`, or ignore an
      existing awake ranges file with `--ignore-awake-ranges`.

    - Sojourns/SIP annotated data, indicating bout boundaries and second-by-
      second estimated metabolic activity. This file is generated by this
      program, but if it already exists it will not be recomputed to save
      time. Editing this file by hand is not recommended.

      Example file name: "007/ActiGraph/James_1secDataTable_with_activpal_sojourns.csv"

      By default, this path will be the same as the ActiGraph data with
      "_sojourns" or "_with_activpal_sojourns" added before the ".csv",
      depending on whether activPAL data have been provided.

      Set this with `--soj-path "your_path_here.csv"`.

    Output files:

    - Sojourns/SIP processed data, containing loads of summary measures
      generated from the metabolic estimates. See the README for a detailed
      description of the contents of this file.

      Example file name: "007/ActiGraph/James_1secDataTable_with_activpal_sojourns_processed.csv"

      This will always use the Sojourns/SIP file path with "_processed"
      added before the ".csv".

    """)    # TODO: summary measures
    parser = OptionParser(usage=usage, description=description)
    parser.add_option('-s', '--subject', dest='subj',
                      help='Subject identifier. This tag will be embedded in '
                           'the processed output; the default value is the '
                           'name of the subject directory ("007" in the '
                           'examples).')
    parser.add_option('--ag-path', type='path',
                      help='Path to ActiGraph 1secDataTable data')
    parser.add_option('--ap-path', type='path',
                      help='Path to activPAL Events data')
    parser.add_option('--soj-path', type='path',
                      help='Path to Sojourns/SIP preprocessed ActiGraph data, '
                           'or output path for same')
    parser.add_option('--awake-path', type='path',
                      help='Path to "awake ranges" file in case autodetection '
                           'of non-wear time is poor, or output path for same')
    parser.add_option('--soj-intermediate-path', type='path',
                      help=SUPPRESS_HELP)
    parser.add_option('--ignore-awake-ranges', action='store_true',
                      help='Flag to ignore "awake ranges" file')
    (opts, args) = parser.parse_args()
    if args:
        try:
            path, = map(Path, args)
        except ValueError:
            path = None
        else:
            if not opts.subj:
                opts.subj = path.resolve().parts[-1]
            if not opts.ag_path:
                opts.ag_path = ActiGraphDataTable.sniff(path, epoch=Second())
            if not opts.ap_path:
                opts.ap_path = ActivPALData.sniff(path)
            if not opts.soj_path:
                opts.soj_path = SojournsData.sniff(path)
            if not opts.awake_path:
                opts.awake_path = AwakeRanges.sniff(path)
    elif not any(opts.__dict__.values()):  # This can't be the right way
        parser.print_help()
        parser.exit()
    return opts

if __name__ == '__main__':
    # FIXME mess
    import sys

    opts = parse_options()
    subj = opts.subj
    ag_path = opts.ag_path
    ap_path = opts.ap_path
    soj_path = opts.soj_path
    awake_path = opts.awake_path
    ignore_awake_ranges = opts.ignore_awake_ranges

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
        awake_ranges = AwakeRanges.from_file(awake_path)
    else:
        awake_ranges = None
    soj = SojournsData.from_file(soj_path, awake_ranges)
    # XXX HACK
    if 'steps' not in soj.raw_data.columns:
        ag = ActiGraphDataTable.from_file(ag_path, awake_ranges)
        soj.raw_data['steps'] = ag.raw_data['Steps']
    soj.process()
    # FIXME integrate this better
    if opts.soj_intermediate_path:
        soj.data.iloc[:-1].to_csv(str(opts.soj_intermediate_path))
        contiguous_apply(
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
            str(opts.soj_intermediate_path.with_name(
                    opts.soj_intermediate_path.name.replace(
                        opts.soj_intermediate_path.suffix,
                        '_squashed'+opts.soj_intermediate_path.suffix))))
    if awake_path and not awake_path.exists():
        soj.awake_ranges.to_file(awake_path)
    summary = ActivitySummary.from_sojourns(soj, subj)
    outpath = soj_path.with_name(soj_path.stem + '_processed.csv')
    summary.to_file(outpath)
