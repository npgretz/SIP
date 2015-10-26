SIP: Analysis of ActiGraph and activPAL accelerometer data
==========================================================

SIP (Sojourns Including Posture) is a method to estimate activity intensity
from ActiGraph and activPAL accelerometer data.  See:

> Ellingson LD, Schwabacher IJ, Kim Y, Welk GJ, Cook DB.
> _Criterion Validity of SIP, an integrative method for processing physical
> activity data._
> Manuscript in preparation.

It is an extension of the Sojourns method presented in this paper:

> [Lyden K, Keadle SK, Staudenmayer J, Freedson PS.
> A method to estimate free-living active and sedentary behavior from an
> accelerometer.
> _Med. Sci. Sports Exerc._
> 2014;46(2):386–97.](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4527685/)

Installation
------------

### For newbies ###

Read INSTALL.md for *much* more detail on how to install SIP on OS X, which is
the system I recommend for non-gurus. There are tentative installation
instructions for Windows, but if you make it all the way through them you will
probably be using the "for experts" instructions from then on.

### For experts ###

To install, clone this repository (or just download and extract the zip archive
version) and add it to your path. SIP isn't packaged for distribution over PyPI
or homebrew, so you'll also need to install this pile of dependencies:

- Python 2.7 (neither 2.6 nor 3.x will work)
- python packages:
  - matplotlib
  - nose (if you want to run the tests)
  - numpy
  - pandas
  - pathlib
- R (works on 3.2.2; probably not hugely sensitive to version)
- R libraries:
  - nnet
  - zoo

Sorry about that.

Method
------

The original Sojourns method takes as input ActiGraph accelerometer data
integrated into "counts" per one-second epoch and annotates this input with a
decomposition of the time series into bouts of consistent activity and
estimates of the metabolic intensity of each bout. It accomplishes this by a
three-stage process: first, it partitions the input into candidate bouts and
merges neighboring bouts until it is satisfied; second, it uses a neural net to
split bouts into four categories (sedentary, lifestyle, locomotion and sport),
assigning intensities to the former two categories based on simple criteria;
and third, it uses a second neural net to estimate the intensities of bouts
from the latter two categories.

SIP modifies this by using concurrently-collected activPAL accelerometer data
in the first step to generate additional candidate bout boundaries whenever the
activPAL detected posture changes, and in the second to determine whether low-
intensity activity was seated or upright (corresponding to the sedentary or
lifestyle categories, respectively), as the thigh-mounted activPAL is known to
be superior to the hip-mounted ActiGraph at making this determination.

What's in this repository?
--------------------------

### The R stuff ###

First, there is sip.functions.R, which is the R code implementing the SIP
method itself, and the various .RData files containing the model parameters
estimated by Lyden, et al. If you know R and want to do everything
interactively at the R prompt, you can:

    # load the code and models
    load("your.path.here/sip/nnet3ests.RData")
    load("your.path.here/sip/cent.1.RData")
    load("your.path.here/sip/scal.1.RData")
    load("your.path.here/sip/class.nnn.use.this.RData")
    source("your.path.here/sip/sip.functions.R")
    library(nnet)

    # read input data
    actigraph <- AG.file.reader("your.input.file.here")

    # leaving out these two steps *should* give the same output as plain
    # sojourns; if it doesn't, please report it as a bug
    activpal <- AP.file.reader("your.activPAL.file.here")
    data <- enhance.actigraph(actigraph, activpal)

    # run SIP
    sip.estimate <- sojourn.3x(data)

    # play with sip.estimate here

    # save the SIP output to disk
    sojourns.file.writer(sip.estimate, "your.output.file.name.here")

For comparison, Lyden, et al. have made their source code available
[here](http://www.math.umass.edu/~jstauden/SojournCode.zip).

### The python stuff ###

There are two python programs in the repository. The first is sip.py, which is
a command-line wrapper around sip.functions.R that post-processes the output
into a CSV file with a large number of summary measures. Each day gets its own
row in the CSV file. See *TODO* below for an explanation of the various summary
measures.

This program also uses a simple heuristic (60 minutes with no ActiGraph counts)
to estimate when the subject was not wearing the monitor, and writes a CSV file
containing the list of guessed wear-time intervals. By modifying this file, you
can cause sip.py and display\_sojourns.py to include or exclude time intervals
according to your subject's wear-time log.

The second program is display\_sojourns.py, which displays graphs of ActiGraph
data at 1- or 60-second epochs, activPAL data, and processed Sojourns/SIP
output. ActiGraph data at 1-second epochs is plotted as raw counts; activPAL
data are plotted as bars indicating when the subject was sitting, standing or
stepping, and the remaining formats are plotted as bars indicating estimated
activity intensity.

The python programs have similar interfaces, and you can get help by running

    sip.py --help
    display_sojourns.py --help

at the command-line. If each subject's data are placed in a directory structure
like the one in the `sampledata/` directory of this repository, you can run
both of these programs simply by giving the name of the subject's directory as
the sole argument. In order for this to work, your input files will need to
have the same extensions that they are given by default when downloading from
their respective devices, and your subdirectories must be named exactly
"ActiGraph" and "activPAL". You can also omit the subdirectories and just place
all files directly in the subject's directory, if you prefer.

If for some reason you need to clean the data by hand, you can save the cleaned
data to a file with a name ending in "\_QC.csv", and the python programs will
select that file preferentially over one without that suffix.

Additionally, there is a test suite in test.py, but it only covers a small
fraction of the code. You can run the test suite with

    nosetests sip

but a better test might just be to run both programs with the sample data:

    ./sip.py sampledata
    ./display_sojourns.py sampledata

Summary measures from sip.py
----------------------------

Note that the various means are weighted by duration, so they answer questions
about the average minute of the sample, not the average bout.

*TODO*
