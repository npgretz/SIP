# sip_utils.py - utilities for accelerometer data processing
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
from __future__ import division

import re
import itertools as it
import functools as ft
import operator as op
import subprocess as sp
import collections

import numpy as np
import pandas as pd

from pandas.tseries.offsets import Day, Minute
from pandas.tseries.frequencies import to_offset

__all__ = []

def export(obj, __all__=__all__):
    __all__.append(obj if isinstance(obj, str) else obj.__name__)
    return obj

@export
def boolify(array, nan=False):
    temp = array.astype(bool)
    if isinstance(temp, pd.core.generic.NDFrame):
        temp[array.isnull()] = nan
    else:
        temp[~np.isfinite(array)] = nan
    return temp

@export
# from subprocess32
def patch_subprocess():
    if hasattr(sp.Popen, '__exit__'):
        return

    def _enter_(self):
        return self

    def _exit_(self, type, value, traceback):
        if self.stdout:
            self.stdout.close()
        if self.stderr:
            self.stderr.close()
        if self.stdin:
            self.stdin.close()
        self.wait()

    sp.Popen.__enter__ = _enter_
    sp.Popen.__exit__ = _exit_

@export
# from itertools cookbook
def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    in_seen = seen.__contains__
    if key is None:
        for element in it.ifilterfalse(in_seen, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

tz = 'America/Chicago'

# ActiGraph cut points based on Freedson, 1998
count_bins = [-np.inf, 0, 100, 760, 1952, 5725, np.inf]
met_bins = [-np.inf, 0, 1.5, 3, 4.5, 6, np.inf]
activity_labels = 'error standing light adl freedson vigorous'.split()
ap_mod_activity_labels = ('error sedentary sitting_active adl_apsed '
                          'freedson_apsed vigorous_apsed'.split())

WEEKDAYS = np.array('Mon Tue Wed Thu Fri Sat Sun'.split())
dtfmt = '%a, %d %b %Y %H:%M:%S %Z %z'

colors = pd.Series(['DarkBlue', 'Black',
                    'Green', 'LightYellow', 'Yellow', 'Orange', 'Red',
                    'Blue', 'Cyan', 'Yellow', 'Orange', 'Red'],
                   index=['sleep']+activity_labels+ap_mod_activity_labels[1:])

class Classifier(object):
    def __init__(self, name, compute):
        self.name = name
        self.cname = '_classifier_' + name
        self.compute = compute.__get__(self)

    def __and__(self, other):
        def compute(this, data):
            if this.cname in data.columns:
                return
            self.compute(data)
            other.compute(data)
            data[this.cname] = data[self.cname] + data[other.cname]
            data.ix[data[this.cname].notnull(),this.cname] = \
                contiguous(data.ix[data[this.cname].notnull(),this.cname])
        return type(self)('%s_and_%s' % (self.name, other.name), compute)

    def allow_breaks(self, dur):
        dur = pd.Timedelta(dur)  # TODO: delete this?
        def compute(this, data):
            if this.cname in data.columns:
                return
            self.compute(data)
            # NOTE: turns 11122x33xx4 into 11111x11xx2, not 11122x22xx3
            data[this.cname] = contiguous(contiguous_apply(
                data, data[self.cname], lambda val, block:
                    pd.Series(block['Dur'].sum() <= dur or not np.isnan(val),
                index=block.index)), asbool=True).where(
                data[self.cname].notnull())
        return type(self)('%s_allow_%d' % (self.name,
                                           dur / pd.Timedelta(1, 'm')),
                          compute)

    def require_duration(self, dur):
        dur = pd.Timedelta(dur)  # TODO: delete this?
        def compute(this, data):
            if this.cname in data.columns:
                return
            self.compute(data)
#            temp = contiguous(boolify(
#                (data.groupby(self.cname)['Dur'].sum() >= dur)[
#                    data[self.cname]]), asbool=True)
#            temp.index = data.index
#            data[this.cname] = temp
            data.ix[boolify(data[self.cname]),this.cname] = contiguous(
                data.ix[boolify(data[self.cname]),self.cname],
                contiguous_apply(
                    data[boolify(data[self.cname])],
                    data.ix[boolify(data[self.cname]),self.cname],
                    lambda val, block:
                        pd.Series(val and block['Dur'].sum() >= dur,
                                  index=block.index)))
        return type(self)('%s_req_%d' % (self.name,
                                         dur / pd.Timedelta(1, 'm')),
                          compute)

    @classmethod
    def from_levels(cls, name, levels):
        def compute(this, data):
            if this.cname in data.columns:
                return
            data[this.cname] = contiguous(data['ActivityLevel'].isin(levels),
                                          asbool=True)
            data.ix[-1,this.cname] = np.nan
        obj = cls(name, compute)
        return obj

    @classmethod
    def from_column(cls, name):
        def compute(this, data):
            if this.cname in data.columns:
                return
            data.ix[:-1,this.cname] = contiguous(data.ix[:-1,this.name],
                                                 asbool=True)
        return cls(name, compute)

    @classmethod
    def from_levels_and_dur(cls, name, levels, dur):
        out = cls.from_levels(name, levels)
        return out.require_duration(dur) if dur else out

# NOTE: 'error' means that the neural net for some reason decided that a bout
#       had a negative intensity.  While this is obviously a problem, it can
#       only happen when the program has already decided that the bout was
#       active.  I hesitate to include such time points in 'pa_recommendation'
#       though.
classifiers = map(Classifier.from_levels, *zip(*[
    ['valid', ['sleep'] + activity_labels + ap_mod_activity_labels],
    ['awake', activity_labels + ap_mod_activity_labels],
    ['sedentary', ['sedentary']],
    ['sitting_active', ['sitting_active']],
    ['standing', ['standing']],
    ['adl_apsed', ['adl_apsed']],
    ['freedson_apsed', ['freedson_apsed']],
    ['vigorous_apsed', ['vigorous_apsed']],
    ['active', activity_labels + ap_mod_activity_labels[2:]],
    ['light', ['sitting_active', 'light', 'standing']],
    ['adl', ['adl', 'adl_apsed']],
    ['freedson', ['freedson', 'freedson_apsed']],
    ['matthews', ['adl', 'adl_apsed', 'freedson', 'freedson_apsed']],
    ['vigorous', ['vigorous', 'vigorous_apsed']],
    ['pa_recommendation', activity_labels[3:] + ap_mod_activity_labels[3:]],
    ['error', ['error']],
]))

classifiers[-2] = classifiers[-2].allow_breaks(pd.Timedelta(2, 'm')) \
    .require_duration(pd.Timedelta(10, 'm'))
classifiers[-2].name = 'pa_recommendation'

classifiers.extend([classifier & classifiers[-2] for classifier in
                    [classifiers[10], classifiers[11], classifiers[13]]])

sed_conv = pd.Series(dict(zip(activity_labels, ap_mod_activity_labels)))

for ident in ['tz', 'count_bins', 'met_bins', 'activity_labels',
              'ap_mod_activity_labels', 'WEEKDAYS', 'dtfmt', 'colors',
              'classifiers', 'sed_conv']:
    export(ident)


def contiguous(data, crit=None, asbool=False):
    if crit is None:
        crit = boolify(data) if asbool else data.notnull()
    else:
        crit = boolify(crit)    # for convenience
    return (data != data.shift()).where(crit).cumsum().where(crit)

@export
def group_by_classifier(data, classifier):
    if classifier.cname not in data.columns:
        classifier.compute(data)
    temp = contiguous_apply(
        data, data[classifier.cname], lambda val, block:
            pd.DataFrame({'counts': block['counts'].sum(),
                          'Dur': block['Dur'].sum(),
                          classifier.cname: val},
                         index=[block.index[0]]))
    return temp.merge(pd.DataFrame(
        # astype for the empty case dammit
        {'bout_Dur': temp.groupby(classifier.cname)['Dur'].sum()
             .astype('m8[ns]')}),
        left_on=classifier.cname, right_index=True, how='left')

@export
def contiguous_apply(data, crit, fun):
    # FIXME: If pandas gets a contiguous groupby, replace this.
    # This is awkward but avoids dropping initial NaNs
    idx = data[(crit != crit.shift()) &
                   ~(crit.isnull() & boolify(crit.isnull().shift()))].index
    out = []
    if len(idx) == 0:
        return fun(np.nan, data)
    elif len(idx) == 1:
        return fun(crit.iloc[0], data)
    for a, b in zip(idx[:-1], idx[1:]):
        out.append(data.loc[a:b].iloc[:-1])
    out.append(data.loc[b:])
    return pd.concat(map(fun, crit[idx], out))


@export
class Sniffable(object):
    @classmethod
    def sniff(cls, dirpath, *args, **kwargs):
        try:
            return cls.sniff_iter(dirpath, *args, **kwargs).next()
        except StopIteration:
            return None

    @classmethod
    def sniff_iter(cls, dirpath, *args, **kwargs):
        return unique_everseen(f for g in cls.search_path(*args, **kwargs)
                                 for f in dirpath.glob(g))

@export
class AwakeRanges(Sniffable):
    def __init__(self, data):
        self.data = data

    def to_series(self, index):
        awake = self.data.reindex(index[:-1], method='ffill', fill_value=False)
        return awake

    def to_file(self, filepath):
        if not hasattr(filepath, 'write'):
            filepath = str(filepath)
        # FIXME: this will not work until pd.DataFrame.to_csv stops forgetting
        # time zones.
        temp = self.data.index.map(op.methodcaller('strftime', dtfmt))
        np.savetxt(filepath, temp.reshape((-1, 2)), fmt='"%s"', delimiter=',')

    @classmethod
    def from_file(cls, filepath):
        if not hasattr(filepath, 'read'):
            filepath = str(filepath)
        # FIXME read_csv currently does bad things to tz abbreviations
        # This horrible hack brought to you by tzlocal()
        temp = pd.read_csv(filepath, names=['wake', 'sleep'])
        idx = pd.DatetimeIndex([
            pd.Timestamp(re.sub(' [A-Z]+ ', ' ', s)).tz_convert(tz)
            for s in temp.values.flat])
        if not idx.is_monotonic:
            raise ValueError("Times in awake ranges file aren't strictly "
                             'increasing!')
        return cls(pd.Series([True, False]*(len(idx)//2), index=idx))

    @classmethod
    def from_series(cls, series):
        temp = series != series.shift()
        # it's very frustrating that bool(NaN) == True
        return cls(boolify(series[temp][boolify(series[temp]) |
                                        boolify(series.shift()[temp])]))

    @staticmethod
    def search_path():
        return ['ActiGraph/*awake ranges_QC.csv',
                'activPAL/*awake ranges_QC.csv',
                '*awake ranges_QC.csv',
                'ActiGraph/*awake ranges.csv',
                'activPAL/*awake ranges.csv',
                '*awake ranges.csv']

@export
class ActivityMonitorData(Sniffable):
    """
    Base class for activity monitor time series data.

    """
    pass

class ActiGraphData(ActivityMonitorData):

    def __init__(self, raw_data, awake_ranges=None):
        self.raw_data = raw_data
        self.awake_ranges = awake_ranges

    # FIXME: Way too much coupling here.
    def process(self):
        self.raw_data['awake'] = self.find_sleep()
        if not self.awake_ranges:
            self.awake_ranges = AwakeRanges.from_series(self.raw_data['awake'])
        self.data = self._process_data()
        self.data['ActivityLevel'], self.data['sedentary'] = \
            self.compute_activity_levels()

    def awake_ranges(self):
        return AwakeRanges.from_series(self.find_sleep())

    def find_sleep(self):
        """
        Find sleep times, either by reading times from a file, or by computing

        """
        if 'awake' in self.raw_data.columns:
            awake = self.raw_data['awake']
        elif self.awake_ranges:
            awake = self.awake_ranges.to_series(self.raw_data.index)
        elif (self.raw_data.ix[:-1,self.sleep_colnames] >= 0).all().all():
            awake = self.infer_sleep()
        else:
            # These times have been manually marked.  Deprecated.
            awake = (self.raw_data.ix[:-1,self.sleep_colnames] >= 0).all(axis=1)
        return awake

    def infer_sleep(self):
        """
        Infer sleep/non-wear time from accelerometer data.

        Assume that any block of duration at least an hour and containing no
        counts is sleep or other non-wear time.  This is not especially robust
        and is virtually guaranteed to miss times when the subject wore the
        monitor while sleeping.

        """
        awake = contiguous_apply(
            self.raw_data.iloc[:-1],
            (self.raw_data.ix[:-1,self.sleep_colnames] > 0).any(axis=1),
            lambda val, block: pd.Series(
                [val or block['Dur'].sum() < pd.Timedelta(1, 'h')],
                index=block.index))
        return awake

    def to_file(self, filepath):
        pass

    def plot_activities(self, start, ax, yinterval):
        dates = days(self.data.index)
        for i, sliced in enumerate(slice_data(self.colors(), dates)):
            time = sliced.index.values - dates[i+1].asm8 + Day().nanos
            ax.broken_barh(np.column_stack([time[:-1],
                                            np.diff(time)]).astype('i8'),
                           yinterval + [i + (dates[0] - start).days, 0],
                           facecolors=sliced.iloc[:-1], edgecolor='none')

    def plot_raw_counts(self, start, ax, yinterval, thr=250):
        dates = days(self.raw_data.index)
        # Guess 250 is a good cutoff, but we'll see
        data = self.raw_data[self.count_colnames].copy()
        data[data > thr] = thr
        to_plot = [yinterval[0]+yinterval[1] - yinterval[1]*
                       data.ix[data[col].diff().diff(-1).astype(bool),col]/thr
                   for col in data.columns]
        for line, color in zip(to_plot, ('red', 'green', 'blue')):
            for i, sliced in enumerate(slice_data(line, dates)):
                sliced.iloc[-1] = sliced.iloc[-2]
                time = sliced.index.values - dates[i+1].asm8 + Day().nanos
                ax.plot(time, sliced + i + (dates[0] - start).days, color)

    def colors(self):
        temp = pd.Series(index=self.data.index)
        temp[:-1] = colors[self.data.ix[:-1,'ActivityLevel']]
        return temp

    def _process_data(self):
        return self.raw_data

    @classmethod
    def from_file(cls, filepath, *args, **kwargs):
        """
        Load an ActiGraph dataset as output from one of the Sojourn programs.

        """
        top_header = cls._slurp_top_header(filepath)
        start_time, epoch = cls._parse_top_header(top_header)
        data = pd.read_csv(str(filepath), skipinitialspace=True,
                           skiprows=len(top_header)+1)
        # append a row of NaNs to mark the end of the last period
        # FIXME: this strategy is sloooow.
        # PERF: This is the #2 perf issue, after sip.R itself
        data = data.append([[]])
        data.index = pd.date_range(start_time, periods=len(data), freq=epoch)
        data.ix[:-1,'Dur'] = np.diff(data.index.values)
        return cls(data, *args, **kwargs)

    @staticmethod
    def _slurp_top_header(filepath):
        isheader = re.compile('[^-,\r\n]').search
        with filepath.open() as f:
            top_header = [l.rstrip('\r\n,') for l in it.takewhile(isheader, f)]
        return top_header

    @staticmethod
    def _parse_top_header(top_header):
        # FIXME: this could use some cleanup
        # Unfortunately, ActiGraph doesn't write this important information in
        # an easily machine-readable format, so break out the fragile regexes.
        metadata = re.match(r"""(?:(?:Start\ Date\ (.*) |
                                      Start\ Time\ (.*) |
                                      Epoch\ Period\ \(hh:mm:ss\)\ (.*) |
                                      .*)\n)*""",
                            '\n'.join(top_header), re.M | re.X).groups()
        start_time = pd.Timestamp(' '.join(metadata[:2]), tz=tz)
        epoch = '{}H{}T{}S'.format(*metadata[2].split(':'))
        return start_time, epoch

@export
class SojournsData(ActiGraphData):
    """
    Sojourns-processed ActiGraph data.

    """

    count_colnames = ['counts', 'counts.2', 'counts.3']
    sleep_colnames = count_colnames

    def _fix_sojourns(self):
        # FIXME: this is a hack
        self.raw_data.ix[:-1,'sojourns'] = \
            contiguous(self.raw_data['sojourns'].where(
                self.raw_data['awake'], -1)).iloc[:-1]

    def compute_activity_levels(self):
        activity_level = pd.Series(index=self.data.index)
        activity_level[:-1] = pd.cut(self.data.ix[:-1,'METs'], met_bins,
                                     right=True, labels=activity_labels)
        activity_level[~boolify(self.data['awake'], True)] = 'sleep'
        try:
            sedentary = self.data['sedentary'].copy()
        except KeyError:
            # can't tell standing from sitting w/o activPAL
            sedentary = pd.Series(index=self.data.index)
            sedentary[:-1] = boolify(activity_level[:-1] == 'standing')
        else:
            sedentary[~boolify(self.data['awake'], True)] = False
        # not sure if boolify is necessary
        activity_level[boolify(sedentary)] = \
            sed_conv[activity_level[boolify(sedentary)]].values
        return activity_level, sedentary

    def plot(self, start, ax, yinterval):
        self.plot_activities(start, ax, yinterval)

    def _process_data(self):
        self._fix_sojourns()
        # Can't use data.groupby because it clobbers the indices!
        return contiguous_apply(self.raw_data, self.raw_data['sojourns'],
                                self.process_one_sojourn)

    @staticmethod
    def process_one_sojourn(idx, block):
        columns = ['counts', 'sojourns', 'METs', 'Dur', 'awake']
        data = [[block['counts'].sum(),
                 idx,
                 block.ix[0,'METs'],
                 block['Dur'].sum(), # Series(dtype='m8').sum() issue
                 block.ix[0,'awake']]]
        # FIXME: this is a terrible hack to check whether there's activPAL data
        if 'ActivityCode' in block.columns:
            columns.append('sedentary')
            data[0].append((block['ActivityCode'] == 0).mean() >= 0.5)
        return pd.DataFrame(data, columns=columns, index=[block.index[0]])

    @staticmethod
    def search_path(with_activpal=None):
        return (['ActiGraph/*_with_activpal_sojourns.csv',
                 '*_with_activpal_sojourns.csv']
                    if with_activpal or with_activpal is None else []) + \
               (['ActiGraph/*_sojourns.csv',
                 '*_sojourns.csv']
                    if not with_activpal or with_activpal is None else [])

@export
class ActiGraphDataTable(ActiGraphData):
    """
    ActiGraph data, integrated into epochs.

    """

    count_colnames = ['Axis1', 'Axis2', 'Axis3']
    sleep_colnames = count_colnames[0:1]

    def _trim_raw_data(self):
        temp = self.raw_data['Axis1'].isnull()
        self.raw_data = self.raw_data[
            self.raw_data.index[~temp][0] :
            self.raw_data.index[~temp.shift().astype(bool)][-1]]
        self.raw_data.iloc[-1] = np.nan

    def plot(self, start, ax, yinterval):
        if self.raw_data.index.freq == Minute():
            self.plot_activities(start, ax, yinterval)
        else:
            self.plot_raw_counts(start, ax, yinterval)

    def colors(self):
        temp = self.data['ActivityLevel']
        temp = temp[temp != temp.shift()]
        temp[:-1] = colors[temp[:-1]]
        return temp

    def compute_activity_levels(self):
        if self.raw_data.index.freq != Minute():
            #FIXME: if freq | Minute(), we should resample.
            raise ValueError("Activity cut points haven't been validated for "
                             'epoch lengths other than 60s')
        activity_level = pd.Series(index=self.data.index)
        activity_level[:-1] = pd.cut(self.data.ix[:-1,'Axis1'], count_bins,
                                     right=False, labels=activity_labels)
        activity_level[~boolify(self.data['awake'], True)] = 'sleep'
        # can't tell standing from sitting w/o activPAL
        activity_level[activity_level == 'standing'] = 'sedentary'
        sedentary = pd.Series(index=self.data.index)
        sedentary[:-1] = boolify(activity_level[:-1] == 'sedentary')
        return activity_level, sedentary

    @classmethod
    def from_file(cls, filepath, *args, **kwargs):
        temp = super(ActiGraphDataTable, cls).from_file(filepath,
                                                        *args, **kwargs)
        temp._trim_raw_data()
        return temp

    @staticmethod
    def search_path(epoch=Minute()):
        if not isinstance(epoch, collections.Iterable):
            epoch = [epoch]
        return [g for e in epoch for g in [
                    'ActiGraph/*%dsecDataTable_QC.csv' % (e.nanos / 1e9),
                    'ActiGraph/*%dsecDataTable.csv' % (e.nanos / 1e9),
                    '*%dsecDataTable_QC.csv' % (e.nanos / 1e9),
                    '*%dsecDataTable.csv' % (e.nanos / 1e9),

@export
class ActivPALData(ActivityMonitorData):
    """
    activPAL event data.

    """
    def __init__(self, data):
        self.data = data

    def to_file(self, filepath):
        pass

    def colors(self):
        temp = contiguous_apply(self.data, self.data['ActivityCode'],
                                lambda val, block:
                                    pd.Series([val], index=block.index[0:1]))
        temp[:] = pd.Series(['Blue', 'Green', 'Red'])[temp].values
        return temp

    def plot(self, start, ax, yinterval):
        dates = days(self.data.index)
        for i, sliced in enumerate(slice_data(self.colors(), dates)):
            time = sliced.index.values - dates[i+1].asm8 + Day().nanos
            ax.broken_barh(np.column_stack([time[:-1],
                                            np.diff(time)]).astype('i8'),
                           yinterval + [i + (dates[0] - start).days, 0],
                           facecolors=sliced.iloc[:-1], edgecolor='none')

    @classmethod
    def from_file(cls, filepath):
        """
        Load the activPAL dataset.

        """
        defpath = filepath.with_name(
            filepath.name.replace(' Events.csv', '.def'))
        defs = cls._load_defs(defpath)
        start_time, stop_time, epoch_nanos = cls._parse_defs(defs)

        data = pd.read_csv(str(filepath))
        # activPAL events file has crufty colnames
        data = data.rename(columns=lambda s: re.sub(r'\s+|\(.*', '', s))
        data = data[boolify(data['Interval'])]
        # This makes me nervous because numpy timedelta handling is shaky.
        # Test this near DST boundaries!
        deltas = (data['DataCount'].values * epoch_nanos).astype('m8[ns]')
        idx = pd.DatetimeIndex(start_time.asm8 + deltas,
                               tz='utc').tz_convert(tz)
        last_interval = data['Interval'].iloc[-1]
        last_interval = \
            ((last_interval * 1e9 / epoch_nanos).round() *
             epoch_nanos).astype('m8[ns]')
        # Sometimes there is raggedness at the end of the dataset
        if stop_time > idx[-1] or last_interval > 0:
            data = data.append([[]])
            idx = idx.append([[max(stop_time, idx[-1] + last_interval)]])
        else:
            data.iloc[-1] = np.nan
        data.index = idx
        return cls(data)

    @staticmethod
    def _load_defs(defpath):
        return pd.read_csv(str(defpath), header=None, index_col=0)[1]

    @staticmethod
    def _parse_defs(defs):
        # Assume local time correct when device activated
        start_time = pd.Timestamp(defs['StartTime'].strip('#'), tz=tz)
        # Device is naive, so can't assume local time correct at stop
        stop_time = pd.Timestamp(defs['StopTime'].strip('#') +
                                 start_time.strftime('%z')).tz_convert(tz)
        # if sampling period is not a whole number of nanoseconds, BADNESS!
        # could fix but YAGNI
        if 1e9 % int(defs['SamplingFrequency']):
            raise ValueError('Sampling period has fractional nanoseconds!')
        epoch_nanos = int(1e9 / int(defs['SamplingFrequency']))
        return start_time, stop_time, epoch_nanos

    @staticmethod
    def search_path():
        return ['activPAL/* Events.csv',
                '* Events.csv']

@export
def process_actigraph(data):
    return data[np.diff(np.concatenate([[np.nan], np.digitize(
        data['Axis1'], count_bins)])) != 0]

@export
def process_activpal(data):
    # concatenate all the steps into blocks of stepping
    return data[np.diff(np.concatenate([[np.nan], data['ActivityCode']])) != 0]

@export
def days(idx):
    return pd.date_range(idx[0].date(), idx[-1].date()+Day(), tz=tz)

@export
def clock_hours(idx):
    hours = pd.date_range(idx[0].date(),
                          idx[-1].date()+Day(), tz=tz, freq='H')
    hours = hours[[0]].append(hours[1:][boolify(np.diff(hours.hour))])
    return hours
#    left, right = slice_inds(hours, idx[[0, -1]])
#    return hours[left:right]

def slice_inds(idx, interval):
    return [idx[1:].searchsorted(interval[0], 'right'),
            idx[:-1].searchsorted(interval[1], 'left')]

@export
def slice_data(timeseries, intervals):
    # TODO: make this work for Series
    slices = []
    for i in range(len(intervals)-1):
        left, right = slice_inds(timeseries.index, intervals[i:i+2])
        # pad with NaNs without dropping type info
        # right+1 is ok because of idx[:-1] in slice_inds
        data = timeseries.shift().iloc[left:right+1].shift(-1)
        if left >= right:
            data.index = [pd.NaT]
        else:
            # FIXME
            data.index = pd.DatetimeIndex(list(it.chain(
                [max(timeseries.index[left], intervals[i])],
                data.index[1:-1],
                [min(timeseries.index[right], intervals[i+1])])))
        slices.append(data)
    return slices

def convert_excel_datetime(series, tz=None):
    # This method is futile because sometimes we don't have enough resolution
    from dateutil.tz import tzoffset
    epoch = pd.Timestamp('1899-12-30')
    temp = series * (24*60*60*1e9)
    drop_factor = 10**(np.log10(temp) - sys.float_info.dig + 1).astype('i8')
    temp = (temp.astype('i8') // drop_factor + 5) // 10 * 10 * drop_factor
    temp = pd.DatetimeIndex(temp.astype('m8').map(epoch.__add__))
    if tz:
        offset = tzoffset(None,
                          temp[0].tz_localize(tz).utcoffset().total_seconds())
        temp = temp.tz_localize(offset).tz_convert(tz)
    return temp
