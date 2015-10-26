#!/usr/bin/env python2

from StringIO import StringIO
from pathlib import Path
import unittest

import numpy as np
from numpy import nan
import pandas as pd
import nose

import sip_utils as util

class TestCase(unittest.TestCase):
    def assert_frame_equals(self, first, second, msg=None):
        """Fail if the two objects are unequal as determined by the .equals
           method.
        """
        assert first.equals(second), msg or "%s not equal to %s" % (first,
                                                                    second)

class TestAwakeRanges(TestCase):

    def setUp(self):
        self.in_csv = StringIO('"Sat, 02 Nov 2013 14:00:00 CDT -0500",'
                                   '"Sun, 03 Nov 2013 01:30:00 CDT -0500"\n'
                               '"Sun, 03 Nov 2013 01:15:00 CST -0600",'
                                   '"Sun, 03 Nov 2013 03:00:00 CST -0600"\n')
        self.out_csv = StringIO()
        self.data = pd.Series([True, False]*2, index=[
            pd.Timestamp(s).tz_convert(util.tz) for s
            in ['2013-11-02 14:00:00 -0500', '2013-11-03 01:30:00 -0500',
                '2013-11-03 01:15:00 -0600', '2013-11-03 03:00:00 -0600']])
        self.idx = pd.date_range('2013-11-02 12:00:00', tz=util.tz,
                                 periods=1501, freq='T')

        self.series = pd.Series(False, index=self.idx[:-1])
        awake_inds = [(120, 810), (855, 960)]
        for wake, sleep in awake_inds:
            self.series[wake:sleep] = True

        self.example = util.AwakeRanges(self.data)

    def tearDown(self):
        self.in_csv.close()
        self.out_csv.close()

    def test_from_file(self):
        data = util.AwakeRanges.from_file(self.in_csv)
        self.assert_frame_equals(data.data, self.data)

    def test_from_series(self):
        data = util.AwakeRanges.from_series(self.series)
        self.assert_frame_equals(data.data, self.data)
        series = pd.Series(index=self.series[:self.data.index[-1]].index)
        series[:-1] = self.series.reindex(series.index[:-1])
        data = util.AwakeRanges.from_series(series)
        self.assert_frame_equals(data.data, self.data)

    def test_to_file(self):
        self.example.to_file(self.out_csv)
        self.assertEqual(self.out_csv.getvalue(),
                                self.in_csv.getvalue())

    def test_to_series(self):
        self.assert_frame_equals(self.example.to_series(self.idx), self.series)


class TestClassifier(TestCase):

    def setUp(self):
        self.idx = pd.date_range('2015-01-01', periods=14, freq='H')
        self.data = pd.DataFrame.from_records(
            [( True,  True,   1,   1, nan,   1,   1,   1,   1,   1,   1, 'u'),
             (False,  True, nan,   1, nan, nan,   1,   1, nan, nan,   1, 'v'),
             (False,  True, nan,   1, nan, nan,   1,   1, nan, nan,   1, 'v'),
             ( True,  True,   2,   1,   1,   2,   1,   1,   2,   2,   1, 'u'),
             ( True,  True,   2,   1,   1,   2,   1,   1,   2,   2,   1, 'u'),
             (False, False, nan, nan, nan, nan, nan, nan, nan, nan, nan, 'v'),
             ( True, False,   3, nan,   2,   2, nan, nan, nan, nan, nan, 'u'),
             ( True,  True,   3,   2,   2,   2, nan,   2,   3,   3,   2, 'u'),
             (False, False, nan, nan, nan, nan, nan, nan, nan, nan, nan, 'v'),
             ( True,  True,   4,   3, nan,   2, nan,   2,   4,   3,   2, 'u'),
             (False, False, nan, nan, nan, nan, nan, nan, nan, nan, nan, 'v'),
             (False, False, nan, nan, nan, nan, nan, nan, nan, nan, nan, 'v'),
             ( True,  True,   5,   4, nan,   3, nan,   3,   5,   4, nan, 'u'),
             (  nan,   nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)],
            index=self.idx,
            columns=['a', 'b', 'x_classifier_a', 'x_classifier_b',
                     'x_classifier_a_req_120', 'x_classifier_a_allow_60',
                     'x_classifier_b_req_120', 'x_classifier_b_allow_60',
                     'x_classifier_a_and_b',
                     'x_classifier_a_allow_60_and_b_allow_60',
                     'x_classifier_b_allow_60_req_120',
                     'ActivityLevel'])
        self.data.ix[:-1,'Dur'] = np.diff(self.data.index.values)
        self.a = util.Classifier.from_column('a')
        self.b = util.Classifier.from_column('b')
        self.a_req_120 = self.a.require_duration(pd.offsets.Hour(2))
        self.b_req_120 = self.b.require_duration(pd.offsets.Hour(2))
        self.a_allow_60 = self.a.allow_breaks(pd.offsets.Hour(1))
        self.b_allow_60 = self.b.allow_breaks(pd.offsets.Hour(1))
        self.a_and_b = self.a & self.b
        self.a_allow_60_and_b_allow_60 = self.a_allow_60 & self.b_allow_60
        self.b_allow_60_req_120 = self.b_allow_60.require_duration(
            pd.offsets.Hour(2))
        self.u = util.Classifier.from_levels('u', ['u'])

    def check_classifier(self, classifier):
        classifier.compute(self.data)
        self.assert_frame_equals(self.data[classifier.cname],
                                 self.data['x' + classifier.cname])

    def test_from_levels(self):
        self.a.compute(self.data)
        self.u.compute(self.data)
        self.assert_frame_equals(self.data[self.a.cname],
                                 self.data[self.u.cname])

    def test_compute(self):
        for classifier in [self.b_allow_60_req_120,
                           self.a_allow_60_and_b_allow_60, self.a_and_b,
                           self.a_allow_60, self.b_allow_60,
                           self.a_req_120, self.b_req_120, self.a, self.b]:
            self.check_classifier(classifier)


class TestClassify(TestCase):

    def setUp(self):
        self.idx = pd.date_range('2014-03-09', '2014-03-10',
                                 tz=util.tz, freq='S')
        self.comp_index = pd.DatetimeIndex(
            ['2014-03-09 00:00:00', # put offsets here once pandas is ready
             '2014-03-09 01:45:00',
             '2014-03-09 01:46:00',
             '2014-03-09 03:30:00',
             '2014-03-09 03:35:00',
             '2014-03-09 03:55:00',
             '2014-03-09 04:00:00',
             '2014-03-09 07:55:00',
             '2014-03-09 08:00:00',
             '2014-03-09 08:30:00',
             '2014-03-09 08:35:00',
             '2014-03-09 08:45:00',
             '2014-03-09 23:00:00',
             '2014-03-09 23:00:01',
             '2014-03-10 00:00:00'], tz=util.tz)

        self.comp_data = pd.DataFrame.from_records(
            [(  0,   0,  1.5,   1,  True),
             (  0,   1,  3.0,   2, False),
             (  0,   0,  4.5,   3,  True),
             (  1,   1,  6.0,   4,  True),
             (  1,   1,  7.5,   5, False),
             (  1,   1,  6.1,   6,  True),
             (  0,   0, -1.0,   7,  True),
             (  0,   0,  4.6,   8, False),
             (  1,   1,  3.1,   9, False),
             (  0,   0,  1.1,  10, False),
             (  0,   0,  1.6,  11,  True),
             (  1,   1, -1.0,  12, False),
             (  0,   1,  3.1,  13,  True),
             (  0,   0,  1.1,  14,  True),
             (nan, nan,  nan, nan,   nan)],
            index=self.comp_index,
            columns=['counts', 'counts.2', 'METs', 'sojourns', 'sedentary'])
        self.comp_data['Axis1'] = self.comp_data['counts']
        self.comp_data['counts.3'] = self.comp_data['counts.2']

        self.data = self.comp_data.reindex(self.idx, method='ffill')
        self.data['Dur'] = np.timedelta64(int(1e9), 'ns')
        self.data.iloc[-1] = nan

        self.awake_ranges = util.AwakeRanges(pd.Series([True, False]*2,
                                                       index=pd.DatetimeIndex(
            ['2014-03-09 03:30:00', '2014-03-09 04:00:00',
             '2014-03-09 08:00:00', '2014-03-09 23:00:01'], tz=util.tz)))

        # 'old' will set df.ix[~df['counts.2'].astype(bool),'Axis1'] = -1
        self.comp_awake = pd.DataFrame.from_records(
            [( False,  False,  False,  False),
             ( False,   True,  False,   True),
             ( False,  False,  False,   True),
             (  True,   True,   True,   True),
             (  True,   True,   True,   True),
             (  True,   True,   True,   True),
             ( False,  False,  False,  False),
             ( False,  False,  False,  False),
             (  True,   True,   True,   True),
             (  True,  False,   True,   True),
             (  True,  False,   True,   True),
             (  True,   True,   True,   True),
             (  True,   True,  False,   True),
             ( False,  False,  False,   True),
             (   nan,    nan,    nan,    nan)],
            index=self.comp_index,
            columns=['awake_ranges', 'old', 'infer_1x', 'infer_3x'])

        self.awake = self.comp_awake.reindex(self.idx[:-1], method='ffill')

        # use 'infer_3x' sleep result
        self.comp_levels = pd.DataFrame.from_records(
            [(         'sleep',     'sleep'),
             (         'light',     'light'),
             (     'adl_apsed',       'adl'),
             ('freedson_apsed',  'freedson'),
             (      'vigorous',  'vigorous'),
             ('vigorous_apsed',  'vigorous'),
             (         'sleep',     'sleep'),
             (         'sleep',     'sleep'),
             (           'adl',       'adl'),
             (      'standing', 'sedentary'),
             ('sitting_active',     'light'),
             (         'error',     'error'),
             (     'adl_apsed',       'adl'),
             (     'sedentary', 'sedentary'),
             (             nan,         nan)],
            index=self.comp_index,
            columns=['with_activpal', 'vanilla'])

        self.levels = self.comp_levels.reindex(self.idx, method='ffill')
        self.levels.iloc[-1] = nan

    def test_find_sleep_infer_1x(self):
        data = util.ActiGraphDataTable(self.data[['Axis1', 'Dur']])
        self.assert_frame_equals(data.find_sleep(),
                                 self.awake['infer_1x'].astype(bool))

    def test_find_sleep_old(self):
        self.data.ix[~self.data['counts.2'].astype(bool),'Axis1'] = -1
        data = util.ActiGraphDataTable(self.data[['Axis1']])
        self.assert_frame_equals(data.find_sleep(),
                                 self.awake['old'].astype(bool))

    def test_find_sleep_awake_ranges(self):
        data = util.SojournsData(self.data[['counts', 'counts.2', 'counts.3']],
                                 self.awake_ranges)
        self.assert_frame_equals(data.find_sleep(),
                                 self.awake['awake_ranges'].astype(bool))

    def test_find_sleep_infer_3x(self):
        data = util.SojournsData(self.data[['counts', 'counts.2', 'counts.3',
                                            'Dur']])
        self.assert_frame_equals(data.find_sleep(),
                                 self.awake['infer_3x'].astype(bool))

    def test_sojourns_classify(self):
        self.comp_data['awake'] = self.comp_awake['infer_3x']
        data = util.SojournsData(None)
        data.data = pd.DataFrame(index=self.comp_index|self.idx[-1:],
                                 columns=['METs', 'awake'])
        data.data.ix[:-1] = self.comp_data[['METs', 'awake']]
        levels, sed = data.compute_activity_levels()
        self.assert_frame_equals(levels, self.comp_levels['vanilla'])

    def test_sip_classify(self):
        self.comp_data['awake'] = self.comp_awake['infer_3x']
        data = util.SojournsData(None)
        data.data = self.comp_data[['METs', 'sedentary', 'awake']]
        levels, sed = data.compute_activity_levels()
        self.assert_frame_equals(levels, self.comp_levels['with_activpal'])
