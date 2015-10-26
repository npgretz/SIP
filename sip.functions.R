acf.lag1 <- function(x)
{

    n <- length(x)
    a <- mean((x[-1]-mean(x[-1]))*(x[-n]-mean(x[-n])))
    v <- var(x)
    if ((v==0)|(is.na(v)))
        val <- 0
    if ((v!=0)&(is.na(v)==F))
        val <- a/v
    return(val)
}

acf.lag1.alt <- function(x)
{
    acf(x, lag.max = 1, plot = FALSE)$acf[2, 1, 1]
}

replace.na <- function(x)
{
    # replace NA values with column means
    col.means <- colMeans(x, na.rm = TRUE)
    col.names <- rep(names(x), each = dim(x)[1])
    x[is.na(x)] <- col.means[col.names[is.na(x)]]
    return(x)
}

find.transitions <- function(y)
{
    N <- length(y)

    trans <- (-diff(y) > 15) & (y[-1] <= 10)
    trans.inds <- c(0,2:N,N)[c(TRUE,trans,TRUE)]
    durations <- diff(trans.inds)

    return(durations)
}

combine.sojourns <- function(durations, short)
{

    # combine too short sojourns.

    # FIXME:
    # I (IJS) think that this and find.transitions() are the weak point of the
    # method. Much improvement could be accomplished by making this smarter.
    # But my efforts to improve it haven't been that effective. If you have
    # lots of free-living training data and want to make SIP/Sojourns better,
    # focus on this!

    # Handle the case where the first or last sojourn is too short
    bool.too.short <- durations<short
    # If all sojourn durations are too short, glom them all.
    if(all(bool.too.short))
        return(sum(durations))
    counter.1 <- which.min(bool.too.short)
    counter.2 <- length(durations)+1-which.min(rev(bool.too.short))
    durations <- c(sum(durations[1:counter.1]),
                   durations[(counter.1+1):(counter.2-1)],
                   sum(durations[counter.2:length(durations)]))

    #   combine too short sojourns with neighboring sojourn.
    #   this loop repeats until there are no more too short sojourns

    repeat
    {
        sojourns <- 1:length(durations)
        too.short <- sojourns[durations<short]
        ts <- length(too.short)

        if(ts==0)
            break

        # now deal with all other too short sojourns
        #   right now i combine too short sojourns with its neighbor that was shorter in duration (e.g. first neighbor = 60 seconds long and second neighbor = 300 seconds long, it gets combined with first neighbor)

        durations.first.neighbors <- durations[too.short-1]
        durations.second.neighbors <- durations[too.short+1]

        too.short.inds.first <- too.short[durations.first.neighbors <=
                                          durations.second.neighbors]
        too.short.inds.second <- too.short[durations.first.neighbors >
                                           durations.second.neighbors]

        sojourns[too.short.inds.first] <- too.short.inds.first-1
        sojourns[too.short.inds.second] <- too.short.inds.second+1

        # deal with instances where need to combine more than 2 sojourns - i.e. short sojourn became first neighbor, and then sojourn before first neighbor also becomes that sojourn via second neighbor grouping - want all 3 of these sojourns to be combined.

        inds.order <- (1:(length(sojourns)-1))[diff(sojourns)<0]
        sojourns[inds.order+1] <- sojourns[inds.order]

        # get new durations now that sojourns are combined 

        durations <- as.vector(tapply(durations,sojourns,sum))

    }

    return(durations)
}

enhance.actigraph <- function(ag,ap)
{
    if(!require(zoo))
        stop("Can't load library zoo!")
    ap$ActivityBlocks <- cumsum(c(TRUE, as.logical(diff(ap$ActivityCode))))
    # It would be nice to leave the datasets as zoo objects, but this seems
    # like it could lead to problems by calling unexpected methods.
    # FIXME need to deal with mismatches in the times spanned by these data
    temp <- merge(zoo(NULL, ag$Time),
                  zoo(ap[c("ActivityCode", "ActivityBlocks",
                           "CumulativeStepCount")],
                      ap$Time - diff(ag$Time)[1]/2))
    temp[1,is.na(temp[1,])] <- 0
    ag[c("ActivityCode", "ActivityBlocks", "AP.steps")] <- na.locf(temp)[ag$Time]

    return(ag)
}

prep.nnetinputs <- function(ag, sojourns, lag.fun)
{
    inputs <- do.call(data.frame, aggregate(ag[1:4], list(sojourns),
        function(x)
        {
            c(X = quantile(x, probs = c(.1, .25, .5, .75, .9)),
              acf = lag.fun(x))
        })[-1])
    # for consistency with the existing data
    names(inputs) <- do.call(paste0, expand.grid(
        c(paste0("X", c(10, 25, 50, 75, 90)), "acf"), ".", c("", 2, 3, "vm")))
    names(inputs)[6] <- "acf"
    inputs$inact.durations <- tapply(sojourns, sojourns, length)
#    # The original code *appears* to replace NAs with column means, but
#    # *actually* the values that would have been NA are initialized to 0 and
#    # their computation is skipped.
    inputs[is.na(inputs)] <- 0
#    inputs[, paste0("acf", c("", ".2", ".3", ".vm"))] <-
#        replace.na(inputs[, paste0("acf", c("", ".2", ".3", ".vm"))])
    return(inputs)
}

sojourn.3x <- function(ag, short = 30)
{
    counts <- ag$counts
    counts.2 <- ag$axis2
    counts.3 <- ag$axis3
    vect.mag <- ag$vm
    y <- counts

    durations <- find.transitions(y)
    durations <- combine.sojourns(durations, short)
    sojourns <- rep(1:length(durations), durations)

    if("ActivityBlocks" %in% colnames(ag))
    {
        temp <- sojourns + ag$ActivityBlocks
        durations <- as.vector(tapply(temp, temp, length))
        durations <- combine.sojourns(durations, short)
        sojourns <- rep(1:length(durations), durations)
    }

    #    make table of durations and sojourns etc

    trans.table <- data.frame(counts = y,
                              counts.2 = counts.2,
                              counts.3 = counts.3,
                              vect.mag = vect.mag,
                              sojourns = sojourns,
                              durations = rep(durations, durations),
                              perc.soj = NA,
                              type = NA,
                              METs = NA,
                              steps = diff(c(0, ag$steps)))

    soj.table <- data.frame(durations = durations,
                            perc.soj = tapply(ag$counts > 0, sojourns, mean),
                            type = 6,
                            METs = NA)

    #   get percent non zero in table


### get inds.inactivities so can test nnet only to distinguish between lifestyle and sedentary

    inputs <- prep.nnetinputs(ag[soj.table$perc.soj[sojourns] < 0.7,],
                              sojourns[soj.table$perc.soj[sojourns] < 0.7],
                              acf.lag1.alt)

    inact.inputs <- as.data.frame(scale(inputs,
                                        center = cent.1,
                                        scale = scal.1))
    rownames(inact.inputs) <- NULL

    #   predict type using all axes + vm.  i intially had a lot of prediction nnets here (ie different axis) but have removed them and only include the one that looks "the best".  there are definitely others we can use/try

    #   remove NaNs created by scaling by 1/0
    inact.inputs <- inact.inputs[,-c(1, 2, 13)]

    #   add soj.type to trans table

    soj.table$type[soj.table$perc.soj < 0.7] <-
        apply(predict(class.nnn.6, inact.inputs), 1, which.max)

#   assign mets to types.

    if("ActivityCode" %in% colnames(ag))
    {
        # bout marked sedentary by activPAL?
        temp <- aggregate(ag$ActivityCode == 0, list(sojourns), mean)$x >= 0.5
        soj.table$type[soj.table$type %in% c(1, 3)] <-
            # 3 if sedentary, 1 if not
            ifelse(temp[soj.table$type %in% c(1, 3)], 3, 1)
    }

    soj.table$METs[(soj.table$type==1)&(soj.table$perc.soj<=0.12)] <- 1.5
    soj.table$METs[(soj.table$type==1)&(soj.table$perc.soj>0.12)] <- 1.7
    soj.table$METs[(soj.table$type==3)&(soj.table$perc.soj<=0.05)] <- 1
    soj.table$METs[(soj.table$type==3)&(soj.table$perc.soj>0.05)] <- 1.2

#   this identifies activities for nnet all - 6 means activity
#   i realize i am getting lag1 differently than i do for inactivities...i should change to use function throughout.
    inputs <- prep.nnetinputs(ag[soj.table$type[sojourns] %in% c(2, 4, 6),],
                              sojourns[soj.table$type[sojourns] %in% c(2, 4, 6)],
                              acf.lag1)
    act.inputs <- inputs[c("X10.","X25.","X50.","X75.","X90.","acf")]
    rownames(act.inputs) <- NULL
    act.inputs <- as.data.frame(scale(act.inputs, center = cent, scale = scal))

#   predict METs

    act.mets.all <- predict(reg.nn,act.inputs)
    soj.table$METs[is.na(soj.table$METs)] <- act.mets.all

#   put back in table

    trans.table$perc.soj <- soj.table$perc.soj[sojourns]
    trans.table$type <- soj.table$type[sojourns]
    trans.table$METs <- soj.table$METs[sojourns]

    trans.table <- trans.table[,-8] # remove $type
    if("ActivityCode" %in% names(ag))
    {
        trans.table$ActivityCode <- ag$ActivityCode
        trans.table$AP.steps <- diff(c(0, ag$AP.steps))
    }
    row.names(trans.table) <- strftime(ag$Time, "%Y-%m-%dT%H:%M:%S%z")
    header <- attr(ag, "AG.header")
    header <- append("Processed with sojourns", header, length(header)-1)
    attr(trans.table, "AG.header") <- header
    return(trans.table)
}   #   end sojourn

sojourns.file.writer <- function(data, filename)
{
    out <- file(filename, open = "w")
    tryCatch(
        {
            writeLines(attr(data, "AG.header"), con = out)
            write.csv(data, file = out)
        }, finally = close(out))
}

AP.file.reader <- function(filename)
{
    # read an activPAL events file.
    opt <- options(stringsAsFactors = FALSE)
    deffile <- sub(" Events\\.csv", ".def", filename)
    header <- read.csv(deffile, header = FALSE, row.names = 1)
    # I refuse to believe that there isn't a better way to do this.
    header <- as.list(as.data.frame(t(header)))

    start.time <- as.POSIXlt(strptime(header$StartTime, "#%Y-%m-%d %H:%M:%S#"))
    samp.freq <- as.numeric(header$SamplingFrequency)

    data <- read.csv(filename)
    # Test whether the timeseries starts at a whole second boundary.
    # This is here because I want to be lazy and see if I need to handle the
    # case where it does not.
    # Update: Never mind, this is useless because the Excel timestamp doesn't
    # always have enough significant figures
#    if(round(data$Time[1] %% 1 * 24*60*60 * samp.freq) %% samp.freq)
#        warning("ActivPAL time series is offset by a fraction of a second.")

    n <- dim(data)[1]
    # We use the start time in the header rather than converting the Excel dates
    # in the data set because Excel pretends DST doesn't exist.
    # This wouldn't work if the data started at a fractional second; hence the
    # test above
    data$Time <- start.time + data$DataCount / samp.freq
    # This should be less fragile but I'm lazy.
    names(data) <- c("Time", "DataCount", "Interval", "ActivityCode",
                     "CumulativeStepCount", "ActivityScore")

    options(opt)
    return(data)
}

AG.file.reader <- function(filename)
{
    # I suspect there's a less verbose way to do this, but I couldn't find it.
    agfile <- file(filename)
    header <- character()
    open(agfile)
    # The regex functions feel like overkill here...
    while(!grepl("^-*$", line <- readLines(agfile, n=1)))
    {
        header <- c(header, line)
        if((temp <- sub("^Start Time ", "", line)) != line)
            start.time <- temp
        if((temp <- sub("^Start Date ", "", line)) != line)
            start.date <- temp
        if((temp <- sub("^Epoch Period \\(hh:mm:ss\\) ", "", line)) != line)
            epoch.period <- temp
    }
    header <- c(header, line)
    start.time <- as.POSIXlt(strptime(paste(start.date, start.time),
                                      "%m/%d/%Y %H:%M:%S"))

    first.line <- readLines(agfile, n=1)
    hasHeader <- substr(first.line, 1, 4) == "Date"
    pushBack(first.line, agfile)

    data <- read.csv(agfile, header = hasHeader)
    close(agfile)

    n <- dim(data)[1]

    if(hasHeader)
    {
        data <- data[,c(paste0("Axis", 1:3), "Vector.Magnitude", "Steps")]
        # Testing this.  Maybe we shouldn't trust the device?
        data$Vector.Magnitude <- sqrt(rowSums(data[,1:3]^2))
    }
    else
    {
        data <- data[,c(1:3, 4, 4)]
        data[,4] <- sqrt(rowSums(data^2))
    }

    names(data) <- c("counts", "axis2", "axis3", "vm", "steps")
    data$steps <- cumsum(data$steps)
    data$Time <- start.time + (0:(n-1))*as.difftime(epoch.period)

    # This is bad form, but works in a pinch
    attr(data, "AG.header") <- header
    return(data)
}

compute.bouts.info <- function(est.mets, units="secs") {
# est.mets is a vector of estimated METs
# units = "secs" or "mins" - the amount of time each entry in est.mets represents
    if(units == "secs") {
        time.units <- 60
    } else {
        time.units <- 1
    }

    mets.length <- length(est.mets)
    inds <- 1:mets.length
    one <- est.mets[-mets.length]
    two <- est.mets[-1]

    # number of transitions from <1.5 to >=1.5
    sed.to.gt.sed.trans <- sum((one<1.5)&(two>=1.5))

    # transitions from <3 to >=3
    trans.up <- (one<3)&(two>=3)

    # transitions from >=3 to <3
    trans.down <- (one>=3)&(two<3)
    trans <- c(0,trans.up+trans.down)
    trans.inds <- (1:mets.length)[trans==1]

    # indices where transitions take place
    trans.inds <- c(1, trans.inds, (mets.length+1))

    # how long are the periods of activity and inactivity
    durations <- trans.inds[-1]-trans.inds[-length(trans.inds)]

    # identify if interval is activity or inactivity (they alternate)
    types <- rep("inactive",length=length(durations))

    if (est.mets[1]<3)
        types <- rep(c("inactive","active"),length=length(durations))
    if (est.mets[1]>=3)
        types <- rep(c("active","inactive"),length=length(durations))

    # Create some empty vectors which will be used to keep track of the
    # start and end points of the bouts in the durations vector.
    bout.starts <- c()
    bout.ends <- c()

    # Bouts can occur in two ways:
    # 1) Multiple periods of >3 MET activity with one or more short periods or low activity in between.
    #    The combined time of low activity is 2 minutes or less and the total time 10 minutes or more.
    # 2) A period of 10 or more uninterrupted minutes of >3 MET activity with large periods of low activity before and after.

    # Search for bouts of the first type:

    # Find all sets of adjacent periods of inactivity with total duration less than 2 minutes.
    indices <- seq_len(length(durations))[types=="inactive"]

    for(i in indices) {
        # amount of inactive time in the current possible-bout
        current.bout.inactive.time <- 0
        # index of the last inactive period that will be included in the current possible-bout
        j <- i

        # add inactive periods to the right of the current starting index of our possible-bout,
        # until adding another would put us over the 2-minute limit
        nextvalue <- durations[i]
        while(current.bout.inactive.time + nextvalue <= 2*time.units) {
            current.bout.inactive.time <- current.bout.inactive.time + nextvalue
            j <- j + 2
            if( j <= length(durations) ) {
                # if we haven't yet reached the end of the durations vector,
                # increment j and get the next value
                nextvalue <- durations[j]
            } else {
                # if we have reached the end of the durations vector,
                # set nextvalue to a large number so we'll exit the loop
                nextvalue <- 2*time.units + 1
            }
        }
        # correct the value of j - we really didn't want to increment it that last time
        # since we ended up not including the corresponding inactive period in our possible-bout.
        j <- j - 2

        # if this possible bout would have already been found by starting from an earlier index, forget about it
        if(i > 2) {
            if(current.bout.inactive.time + durations[i - 2] <= 2*time.units) {
                current.bout.inactive.time <- 0
            }
        }

        # if we found a possible bout, record that information
        if(current.bout.inactive.time > 0) {
            # save the start position of the bout in the durations vector
            # (the bout starts at the period of activity preceeding the period of inactivity located at index i)
            # (unless i = 1, when there is no preceeding period of activity)
            if(i > 1) {
                bout.starts <- c(bout.starts, (i - 1))
            } else {
                bout.starts <- c(bout.starts, 1)
            }

            # save the end position of the bout in the durations vector
            # (the bout ends at the period of activity following the period of inactivity located at index j)
            # (unless j = length(durations), when there is no following period of activity)
            if(j < length(durations)) {
                bout.ends <- c(bout.ends, (j + 1))
            } else {
                bout.ends <- c(bout.ends, j)
            }
        }
    }


    # Out of the possible bouts located above, keep only those with total time of at least 10 minutes.
    keepers <- c()
    for(i in seq_len(length(bout.starts))) {
        if(sum(durations[bout.starts[i]:bout.ends[i]]) >= 10*time.units) {
            keepers <- c(keepers, i)
        }
    }

    bout.starts <- bout.starts[keepers]
    bout.ends <- bout.ends[keepers]


    # Check to see if any of the possible bouts above have overlapping start and end indices.
    # If so, keep the first and eliminate those that overlap with it.
    i <- 1
    while(i < length(bout.starts)) {
        if( bout.starts[i + 1] <= bout.ends[i] ) {
            bout.starts <- bout.starts[-(i + 1)]
            bout.ends <- bout.ends[-(i + 1)]
        } else {
            i <- i + 1
        }
    }



    # Search for bouts of the second type:
    indices <- seq_len(length(durations))[types=="active"]

    for(i in indices) {
        if(durations[i] >= 10*time.units) {
            # Is this a type 2 bout?  it might be..
            is.bout <- TRUE

            # If this period of activity is preceeded by a period of inactivity,
            # check to see how long that period of inactivity was.  If it was short,
            # this is a type 1 bout and will have been located above already
            if(i > 1) {
                if(durations[i - 1] <= 2*time.units) {
                    is.bout <- FALSE
                }
            }

            # If this period of activity is followed by a period of inactivity,
            # check to see how long that period of inactivity was.  If it was short,
            # this is a type 1 bout and will have been located above already
            if(i < length(durations)) {
                if(durations[i + 1] <= 2*time.units) {
                    is.bout <- FALSE
                }
            }

            # If this turned out to be a type 2 bout, add it to bout.starts and bout.ends
            if(is.bout) {
                bout.starts <- c(bout.starts, i)
                bout.ends <- c(bout.ends, i)
            }
        }
    }

    # Convert the values in bout.starts from indices in the durations vector
    # to the corresponding indices in the est.mets vector, and combine the values
    # into one vector to be used to extract the relevant seconds from est.mets
    indices <- c()

    for(i in seq_len(length(bout.starts))) {
        bout.starts[i] <- sum( durations[seq_len( bout.starts[i] - 1 )] ) + 1
        bout.ends[i] <- sum( durations[seq_len( bout.ends[i] )] )
        indices <- c(indices, bout.starts[i]:bout.ends[i])
    }

    num.bouts <- length(bout.starts)
    bout.hours <- length(indices)/(60*time.units)
    bout.MET.hours <- sum(est.mets[indices])/(60*time.units)
    info <- data.frame(num.bouts=num.bouts, bout.hours=bout.hours, bout.MET.hours=bout.MET.hours, sed.to.gt.sed.trans=sed.to.gt.sed.trans)

    return(info)
}


sojourn.1x <- function(counts,perc.cut=0.05,perc.cut.2=0.12,perc.cut.3=0.55,too.short=10,sit.cut=90,long.soj=120)
{

    y <- counts
    # identify sojourns.
    inds <- 1:length(y)

    mmm <- length(y)
    one <- y[-mmm]
    two <- y[-1]

    # transitions from 0 to >0
    trans.up <- (one==0)&(two>0)
    # transitions from >0 to 0
    trans.down <- (one>0)&(two==0)

    trans <- c(0,trans.up+trans.down)
    trans.inds <- (1:mmm)[trans==1]

    # indices where transitions take place
    trans.inds <- c(1,trans.inds,(mmm+1))

    # how long are the sojourns and the zeros
    durations <- trans.inds[-1]-trans.inds[-length(trans.inds)]

    # identify if interval is zeros or >0s (they alternate)
    type <- rep("zeros",length=length(durations))
    if (y[1]==0) 
        type <- rep(c("zeros","act"),length=length(durations))
    if (y[1]>0) 
        type <- rep(c("act","zeros"),length=length(durations))

    soj.table <- data.frame(type,durations,trans.inds=trans.inds[-length(trans.inds)])

    soj.table$act.type.1 <- "undetermined"
    soj.table$act.type.1[(soj.table$type=="zeros")&(soj.table$durations>sit.cut)] <- "sedentary"
    soj.table$act.type.1[(soj.table$type=="act")&(soj.table$durations>too.short)] <- "activity"



    # combine neighboring undetermineds
    mmm <- dim(soj.table)[1]
    prev.was.undet.inds <- 
        (2:mmm)[(soj.table$act.type.1[2:mmm]=="undetermined")&
                    (soj.table$act.type.1[1:(mmm-1)]=="undetermined")]
    if (length(prev.was.undet.inds)>0)
        rev.soj.table <- soj.table[-prev.was.undet.inds,]
    mmm <- dim(rev.soj.table)[1]

    rev.soj.table$durations <- 
        c((rev.soj.table$trans.inds[-1]-
            rev.soj.table$trans.inds[-mmm]),
                rev.soj.table$durations[mmm])

    mmm <- dim(rev.soj.table)[1]

    # find too short undetermineds
    too.short.undet.inds <- (1:mmm)[(rev.soj.table$durations<too.short)&(rev.soj.table$act.type.1=="undetermined")]

    if (length(too.short.undet.inds)>0)
    {   
        while (too.short.undet.inds[1]==1)
        {   
            too.short.undet.inds <- too.short.undet.inds[-1]
            rev.soj.table <- rev.soj.table[-1,]
            rev.soj.table$trans.inds[1] <- 1
            mmm <- dim(rev.soj.table)[1]
            too.short.undet.inds <- too.short.undet.inds-1
        }

        last <- length(too.short.undet.inds)
        while (too.short.undet.inds[last]==mmm)
        {
            too.short.undet.inds <- too.short.undet.inds[-last]
            junk <- rev.soj.table$durations[(mmm-1)]
            rev.soj.table <- rev.soj.table[-mmm,]
            mmm <- dim(rev.soj.table)[1]
            rev.soj.table$durations[mmm] <- junk+rev.soj.table$durations[mmm]
            last <- length(too.short.undet.inds)
        }

        # short undetermineds between two acts of same type
        to.delete.inds <- 
            (too.short.undet.inds)[rev.soj.table$act.type.1[too.short.undet.inds-1]==rev.soj.table$act.type.1[too.short.undet.inds+1]]
        done.inds <- (1:length(too.short.undet.inds))[rev.soj.table$act.type.1[too.short.undet.inds-1]==rev.soj.table$act.type.1[too.short.undet.inds+1]]
        too.short.undet.inds <- too.short.undet.inds[-done.inds]

        # between two acts of different types
        junk <- rev.soj.table[too.short.undet.inds,]

        junk$act.type.1 <- "sedentary"
        junk$act.type.1[junk$type=="act"] <- "activity"
        rev.soj.table[too.short.undet.inds,] <- junk

        rev.soj.table <- rev.soj.table[-to.delete.inds,]


    }


    mmm <- dim(rev.soj.table)[1]
    junk <- c(rev.soj.table$act.type.1[2:mmm]==rev.soj.table$act.type.1[1:(mmm-1)])
    same.as.prev.inds <- (2:mmm)[junk]
    if (length(same.as.prev.inds)>0)
    {
        rev.soj.table <- rev.soj.table[-same.as.prev.inds,]
        mmm <- dim(rev.soj.table)[1]    
        rev.soj.table$durations <- 
            c((rev.soj.table$trans.inds[-1]-
                rev.soj.table$trans.inds[-mmm]),
                    rev.soj.table$durations[mmm])
        last.obs <- rev.soj.table$durations[mmm]-1+rev.soj.table$trans.inds[mmm]

        if (last.obs != length(y))
            rev.soj.table$durations[mmm] <- length(y)-rev.soj.table$trans.inds[mmm]+1

    }

    trans.inds <- c(rev.soj.table$trans.inds,length(y)+1)
    durations <- trans.inds[-1]-trans.inds[-length(trans.inds)]

    soj.table <- data.frame(durations)

    sojourns <- rep(1:length(soj.table$durations),soj.table$durations)
    perc.gt.0 <- tapply(y>0,sojourns,mean)

    soj.table$perc.gt.0 <- perc.gt.0

    soj.table$revised.type <- "sit.still"
    soj.table$revised.type[soj.table$perc.gt.0>perc.cut.3] <- "activity"
    soj.table$revised.type[(soj.table$perc.gt.0>perc.cut)&(soj.table$perc.gt.0<=perc.cut.2)&(soj.table$durations>sit.cut)] <- "sit.move"
    soj.table$revised.type[(soj.table$perc.gt.0>perc.cut)&(soj.table$perc.gt.0<=perc.cut.2)&(soj.table$durations<=sit.cut)] <- "stand.still"
    soj.table$revised.type[(soj.table$perc.gt.0>perc.cut.2)&(soj.table$perc.gt.0<=perc.cut.3)] <- "stand.small.move"

    durations <- soj.table$durations
    type <- soj.table$revised.type

    sojourns <- rep(1:length(durations),durations)
    type <- rep(type,durations)
    perc.gt.0 <- rep(perc.gt.0,durations)
    durations <- rep(durations,durations)
    nnn <- length(sojourns)

    longer.acts <- unique(sojourns[(durations>(long.soj-1))])

    f <- function(s)
    {
        dur <-  unique(durations[sojourns==s])
        sub.sojourns <- rep(1:floor(dur/(long.soj/2)),
            times=c(rep((long.soj/2),floor(dur/(long.soj/2))-1),
            dur-(floor(dur/(long.soj/2))-1)*(long.soj/2)))
        sub.sojourns <- s + sub.sojourns/(max(sub.sojourns)+1)
        return(sub.sojourns)
    }
    new.values <- sapply(longer.acts,f)
    starts <- sapply(match(longer.acts,sojourns),paste0,":")
    ends <- length(sojourns) - match(longer.acts,rev(sojourns)) + 1
    indices <- mapply(paste0,starts,ends,USE.NAMES=F)
    indices <- unlist(lapply(parse(text = indices), eval))
    sojourns[indices] <- unlist(new.values)

    # apply METs to zeros
    METs <- rep(NA,length(type))
    METs[(type=="sit.still")] <- 1
    METs[(type=="sit.move")] <- 1.2
    METs[(type=="stand.still")] <- 1.5
    METs[(type=="stand.small.move")] <- 1.7


    data <- data.frame(counts=y,sojourns=sojourns,durations=durations,type=type,METs=METs,perc.gt.0=perc.gt.0)

    # prepare to apply nnet to the activity sojourns
    nnn <- dim(data)[1]
    act.inds <- (1:nnn)[(data$type=="activity")]
    act.data <- data[act.inds,]
    act.durations <- table(act.data$sojourns)

    quantiles <- tapply(act.data$counts,act.data$sojourns,quantile,p=c(.1,.25,.5,.75,.9))
    nn.data <- as.data.frame(do.call("rbind",quantiles))
    nn.data$acf <- tapply(act.data$counts,act.data$sojourns,acf.lag1)
    nn.data <- nn.data[,c(1:6)]

    names(nn.data) <- c("X10.","X25.","X50.","X75.","X90.","acf")

    nnetinputs <- scale(nn.data,center=cent,scale=scal)

    # apply nnet and put it back into the dataset
    est.mets.1 <- NA #predict(MA.reg.nn,nnetinputs)
    est.mets.2 <- predict(ALL.reg.nn,nnetinputs)

    #act.mets.1 <- rep(est.mets.1,act.durations)
    act.mets.2 <- rep(est.mets.2,act.durations)

    data$METs <- METs
    data$METs.2 <- METs

    data$METs[act.inds] <- act.mets.2
    data$METs.2[act.inds] <- act.mets.2

    data$level <- "sed"
    data$level[data$METs>=1.5] <- "light"
    data$level[data$METs>=3] <- "mod"
    data$level[data$METs>=6] <- "vig"
    data$level <- factor(data$level,levels=c("sed","light","mod","vig"))

    data$level.2 <- "sed"
    data$level.2[data$METs.2>=1.5] <- "light"
    data$level.2[data$METs.2>=3] <- "mod"
    data$level.2[data$METs.2>=6] <- "vig"
    data$level.2 <- factor(data$level.2,levels=c("sed","light","mod","vig"))
    n <- dim(data)[1]
    inds <- (1:n)[data$METs<1]
    data$METs[inds] <- 1

    data <- data[,c(1,2,3,4,5,6,8)]
    data
}   




