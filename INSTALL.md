Installing SIP
==============

Unfortunately, at this time SIP can't be installed directly by a package
manager. If I ever get around to making that possible, this file will get a lot
shorter very quickly.

OS X
----

### Step 0: Terminal ###

All of your interactions with SIP will happen at the terminal prompt, so you
need to open a terminal before anything else. OS X comes with a terminal, but
it's hidden away in Utilities. Go to the Finder and use the menus to navigate:

    Go -> Utilities

From there, drag Terminal.app down to your dock where you can get to it easily.
Now click on the Terminal icon to open a terminal.

### Step 1: Homebrew ###

First, install Homebrew. For this install only, you will need to be logged in
as an administrator. At the bottom of the [Homebrew home page](http://brew.sh)
there is a section on installing Homebrew that tells you to paste a command
into the terminal prompt. Currently, that command is this:

    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

This downloads the installation script and runs it. Hit return each time it
asks you to confirm that it's doing what you want, and enter your password when
it asks you to. It should do this twice; once in the terminal window and once
in a popup when it's installing the XCode command line tools.

Once it's done installing, run

    brew doctor

and observe that "Your system is ready to brew."

### Step 2: Python ###

The next step is to install a recent python, since Apple likes to ship an
outdated or otherwise slightly off-kilter python. Start with this:

    brew install python

which installs not only python but also the python sort-of-package-manager pip.
Follow up with

    pip install pathlib numpy matplotlib nose pandas

to install all of the python packages you'll need.

### Step 3: R ###

To install R, you first need to install XQuartz, since Apple no longer packages
X11 with OS X. Go to the [XQuartz home page](http://xquartz.macosforge.org) and
download and run the installer. Give it your administrator password again, and
after it's done let it log you out and back in.

Now it's time to install R:

    brew tap homebrew/science
    brew install R

To install the necessary packages, fire up R:

    R
    # This stuff is inside R:
    install.packages("zoo")
    install.packages("nnet")
    q()

### Step 4: SIP ###

First, go to the directory you want to install SIP into and clone the Git
repository:

    # Install into your home directory
    # ("~" is a shorthand for /Users/your_name_here)
    cd ~
    git clone https://www.github.com/ischwabacher/SIP.git

This creates a SIP directory `/Users/your_name_here/SIP` containing all the
files you need to run SIP. Next, add SIP to your path so that the shell can
find it:

    # Don't change directories between the previous command and this one
    cat <<EOF >>~/.bashrc
    export SIP_DIR=$(printf %q "$PWD")/SIP
    export PATH=$SIP_DIR${PATH:+:$PATH}
    EOF

    cat <<'EOF' >>~/.bash_profile
    if [[ $- == *i* ]]; then
        . ~/.bashrc
    fi
    EOF

    . ~/.bashrc

And you're done!

As long as you don't modify any of the files in the SIP directory, in the event
that I make improvements to it you can update it by opening a terminal and
running this:

    # The "export SIP_DIR=..." command above lets us just use $SIP_DIR for the
    # SIP directory
    cd "$SIP_DIR"
    git pull

Pretty nifty, huh?

### Step 5: Test it out ###

SIP hasn't been checked over by the kinds of teams that write software for
nuclear reactors and space probes, so it almost certainly has bugs. There isn't
a comprehensive test suite yet, but you can at least check that it runs:

    # Just run Sojourns
    sip.py -s sampledata \
        --ag-path "$SIP_DIR"/sampledata/ActiGraph/sampledata_1secDataTable.csv
    # Run SIP, incorporating activPAL data
    sip.py "$SIP_DIR"/sampledata
    display_sojourns.py "$SIP_DIR"/sampledata

This should give you some nice graphs. If you want to run the test suite, you
can do this:

    nosetests sip

At this stage, the tests barely test anything and you shouldn't put much stock
in the fact that they pass, but over time the test suite may become a better
indicator that everything is indeed working.

Windows
-------

I haven't used Windows in almost a decade, so everything in this section is
based on guessing and internet research. Following these directions will almost
certainly not go smoothly and will require a lot of figuring things out
yourself.

The easiest way *might* be to just cut the Gordian knot and install
[Cygwin](https://www.cygwin.com/). This will give you a UNIX-like environment
to play in on your Windows machine. Just install everything as in the OS X
install, but use Cygwin for any of the steps involving brew or other Mac-
specific tools. The only reason I don't just heartily endorse this plan is that
without brew I have no idea how much tracking down of extra bits you will have
to do. This might be the easiest way, or you might end up hunting for a FORTRAN
compiler in a seedy internet back alley. You really won't know until you try.

If Cygwin doesn't work for you, here's what you'll need:

### Step 1: Python ###

Because Microsoft doesn't make its compilers free the way Apple does (as in
beer) or GNU does (as in speech), and because there's no equivalent to Homebrew
for Windows, installing all of the python bits we need is trickier. I see three
options:

#### Use a prepackaged distribution ####

The SciPy project's [installation guide](http://www.scipy.org/install.html)
suggests using a prepackaged distribution to install everything you need. If
you go this route, be aware that some of these packages have very specific
citation requirements embedded in their licenses (which is why I didn't use
them in the first place).

You will need to make sure that you install Python 2.7, preferably 2.7.9+.
(Python 3 is a slightly different language and sojourns won't work with it
right now.) Install the most recent versions of pathlib, numpy and matplotlib,
and version 0.14.1 of pandas.

#### Use Christoph Gohlke's pre-built packages ####

A good unofficial source for pre-built Windows packages is
[here](http://www.lfd.uci.edu/~gohlke/pythonlibs/). Download
[Python 2.7.10](https://www.python.org/downloads/release/python-2710/), as well
as [nose](http://www.lfd.uci.edu/~gohlke/pythonlibs/#nose),
[pathlib](http://www.lfd.uci.edu/~gohlke/pythonlibs/#backports),
[numpy-MKL](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy),
[python-dateutil](http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-dateutil),
[six](http://www.lfd.uci.edu/~gohlke/pythonlibs/#six),
[pytz](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pytz),
[setuptools](http://www.lfd.uci.edu/~gohlke/pythonlibs/#setuptools),
[pyparsing](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyparsing),
[matplotlib](http://www.lfd.uci.edu/~gohlke/pythonlibs/#matplotlib), and
[pandas](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pandas), making sure each
time to get the `py2.7` version compatible with your computer (`win32` if it's
32-bit; `win-amd64` if it's 64).

You will need to set up your PATH, which is similar in concept to what you have
to do on OS X, but completely different in execution.

#### Use pip ####

The advantage of pip is that it knows how to fetch all of a project's python
dependencies; the downside is that many projects don't expose pre-built
versions in the Python Package Index, so it has to compile things from source.

Microsoft recently released a [free compiler](http://aka.ms/vcpython27) for
building python extensions. If you download it and
[Python 2.7.10](https://www.python.org/downloads/release/python-2710/), you can
try to proceed with the python installation as described in the Mac OS X
section with this command:

    pip install pathlib numpy matplotlib nose pandas

Unfortunately, pip doesn't handle non-python dependencies, so you may find
yourself in the awkward position of having to track down the same kinds of bits
that you presumably already tried and failed to install with Cygwin.

The python ecosystem is evolving in a direction that will lead to more projects
having pre-built Windows binaries in the Python Package Index, which will make
pip correspondingly more useful on Windows, but at this point if this way
doesn't just work, I'd recommend abandoning it and trying one of the other
paths.

### Step 2: R ###

Download and install
[R for Windows](http://cran.r-project.org/bin/windows/base/), then open R and
run this:

    install.packages("zoo")
    install.packages("nnet")
    q()

### Step 3: SIP ###

It's probably easiest to just download a zip archive of SIP from
[here](https://www.github.com/ischwabacher/SIP/archive/master.zip). Unzip the
archive and add it to your PATH. Again, I don't know how to edit the PATH on
Windows, and this is something that should be done by someone who does.

### Step 4: Test it ###

Open a
[Command Prompt](http://windows.microsoft.com/en-us/windows/command-prompt-faq)
(cmd.exe) and do the same things you would do on OS X.

Because the Windows command prompt is different from the bash shell, you will
need to learn how to use it. Here's an [intro](http://dosprompt.info/), and
here's the
[documentation](http://technet.microsoft.com/en-us/library/cc754340.aspx). Keep
in mind that quoting rules are
[different](http://msdn.microsoft.com/en-us/library/a1y7w461.aspx).

Other UNIX-like operating systems
---------------------------------

I haven't tried this, but it should work fine. Replace homebrew with your
favorite package manager and install as with OS X.
