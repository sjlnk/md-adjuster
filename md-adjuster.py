import csv
import os.path
import shutil
import re
import argparse
import logging
import urllib.request
import sys
import numpy as np
import bottleneck as bn
from datetime import datetime, timedelta

def bytes_to_str(b):

    if b <= 1024:
        return "{} B".format(b)
    elif b <= 1024 ** 2:
        return "{} KB".format(round(b/1024, 2))
    elif b <= 1024 ** 3:
        return "{} MB".format(round(b/1024**2, 2))
    elif b <= 1024 ** 4:
        return "{} GB".format(round(b/1024**3, 2))
    elif b <= 1024 ** 5:
        return "{} TB".format(round(b/1024**4, 2))
    else:
        return "{} PB".format(round(b/1024**5, 2))


class DivSplitAdjustment:

    def __init__(self, type, date, adj):
        self.type = type
        self.date = date
        self.adj = adj

    def __repr__(self):
        return "{}: {} adj={}".format(self.date, self.type, self.adj)

def analyze_datafile(file, sensitivity=1e-1):

    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        median_diffs = None
        median_diffs_idx = []
        lineno = 1
        allfloats = []
        for row in reader:
            floats = []
            for i in range(len(row)):
                try:
                    f = float(row[i])
                    floats.append(f)
                    # only on first run
                    if not median_diffs:
                        median_diffs_idx.append(i)
                except ValueError:
                    pass
            # most likely a header line
            if not floats:
                continue
            allfloats.append(floats)
            if not median_diffs:
                median_diffs = [[] for _ in range(len(floats))]
                logging.debug("Possible price indices: {}".format(median_diffs_idx))
            if len(floats) != len(median_diffs):
                logging.error("Line {} in file {} is exceptional, skipped.".format(lineno, file))
                continue
            fmedian = bn.median(floats)
            # logging.debug("[{}] median: {}".format(lineno, fmedian))
            for i in range(0, len(floats)):
                diffnow = (floats[i] - fmedian) / fmedian
                median_diffs[i].append(diffnow)
                # logging.debug("[{}][{}] diff: {}".format(lineno, i, diffnow))
            lineno += 1

            # we don't need more data than this (faster runtime)
            if lineno == 10001:
                break

    mean_median_diffs = np.abs(np.mean(median_diffs, axis=1))
    logging.debug("Mean median diffs: {}".format(mean_median_diffs))

    median_diffs_idx = np.array(median_diffs_idx)
    mean_median_diffs = np.array(mean_median_diffs)

    pricecols = median_diffs_idx[mean_median_diffs < sensitivity]

    numdecimals = 0

    allfloats = np.asarray(allfloats)
    for f in allfloats.flat:
        splitted = str(f).split(".")
        if len(splitted) == 2:
            numdecimals = max(numdecimals, len(splitted[1]))

    # maximum number of decimals is limited
    numdecimals = min(numdecimals, 10)

    return pricecols, numdecimals


def get_dt_format(file):

    dtformats = ["%Y%m%d %H:%M:%S", "%Y%m%d"]

    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        firstline = True
        for row in reader:
            if firstline:
                firstline = False
                continue
            # datetime should be the first cell
            sampledtstr = row[0]
            break

    logging.debug("Sample datetime string: '{}'".format(sampledtstr))

    for dtformat in dtformats:
        try:
            dt = datetime.strptime(sampledtstr, dtformat)
            return dtformat
        except ValueError:
            pass

    logging.error("Datetime format cannot be guessed.")
    return None

def analyze_dates(datafile, dtformat):

    dates = []

    with open(datafile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                # datetime should be first column
                dt = datetime.strptime(row[0], dtformat)
                dates.append(dt)
            except ValueError:
                pass

    if dates[-1] > dates[0]:
        ascending = True
        logging.info("Chronological order of data is ascending.")
    else:
        ascending = False
        logging.info("Chronological order of data is descending.")

    # checking for inconsistencies
    for i in range(1, len(dates)):
        if ascending:
            if dates[i] <= dates[i-1]:
                logging.error("Chronological order of data is inconsistent at line {}.".format(i))
                return None, None, None
        else:
            if dates[i] >= dates[i-1]:
                logging.error("Chronological order of data is inconsistent at line {}.".format(i))
                return None, None, None

    return min(dates), max(dates), ascending


def download_yahoo_data(symbol, startdate, enddate, type, format):

    sd = startdate
    ed = enddate

    if format == "csv":
        urltodl = "http://ichart.finance.yahoo.com/table.csv?"
    elif format == "x":
        urltodl = "http://ichart.finance.yahoo.com/x?"
    else:
        print("format ({}) not supported.".format(format))
        return

    urltodl += "s={symbol}".format(symbol=symbol)
    urltodl += "&a={startm}".format(startm=str(sd.month - 1).rjust(2, "0"))
    urltodl += "&b={startd}&c={starty}".format(startd=str(sd.day).rjust(2, "0"), starty=sd.year)
    urltodl += "&d={endm}".format(endm=str(ed.month - 1).rjust(2, "0"))
    urltodl += "&e={endd}&f={endy}&g={type}".format(endd=str(ed.day).rjust(2, "0"), endy=ed.year, type=type)

    logging.debug("Downloading historical Yahoo data from: {}".format(urltodl))

    req = urllib.request.urlopen(urltodl)
    data = req.read()
    datastr = data.decode()

    return datastr

# data has to be in "x" format
def construct_ds_multipliers(yahoodata):

    lines = yahoodata.splitlines()
    reader = csv.reader(lines)

    # 0 = none, 1 = dividend, 2 = split
    waiting_for = 0

    prev_multiplier = -1

    dsadj = []

    for row in reader:
        try:
            dt = datetime.strptime(row[0], "%Y%m%d")
        except ValueError:
            if row[0] == "DIVIDEND":
                waiting_for = 1
                dsdate = datetime.strptime(row[1].strip(), "%Y%m%d")
            elif row[0] == "SPLIT":
                waiting_for = 2
                dsdate = datetime.strptime(row[1].strip(), "%Y%m%d")
            continue


        c = float(row[4])
        ac = float(row[6])

        if prev_multiplier == -1:
            prev_multiplier = ac / c
            continue

        multnow = ac / c

        if waiting_for == 1:
            if dt < dsdate:
                adj = multnow / prev_multiplier
                assert(adj < 1)
                dsadj.append(DivSplitAdjustment("DIVIDEND", dsdate, adj))
                logging.debug(dsadj[-1])
                waiting_for = 0
        elif waiting_for == 2:
            if dt < dsdate:
                adj = multnow / prev_multiplier
                dsadj.append(DivSplitAdjustment("SPLIT", dsdate, adj))
                logging.debug(dsadj[-1])
                waiting_for = 0

        prev_multiplier = multnow

    return dsadj

def adjust_price_columns(datafile, dsmult, pricecols, ofile, dtformat, numdecimals, mode):

    datarows = []
    datadates = []

    with open(datafile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            datarows.append(row)
            dt = datetime.strptime(row[0], dtformat)
            datadates.append(dt)

    if datadates[-1] > datadates[0]:
        ascending = True
    else:
        ascending = False

    if ascending:
        datarows.reverse()
        datadates.reverse()

    startdsidx = -1

    for i in range(len(dsmult)):
        if dsmult[i].date < datadates[0]:
            startdsidx = i
            break

    if startdsidx == -1:
        logging.error("Data ranges don't match, adjustments not necessary!")
        return False

    nextdsidx = startdsidx

    totdivs = 1
    totsplits = 1

    for i in range(len(datarows)):

        dt = datadates[i]

        if nextdsidx < len(dsmult):
            ds = dsmult[nextdsidx]

            if dt < ds.date:
                if ds.type == "DIVIDEND":
                    totdivs *= ds.adj
                elif ds.type == "SPLIT":
                    totsplits *= ds.adj
                nextdsidx += 1

        for i2 in pricecols:
            datarows[i][i2] = float(datarows[i][i2])
            if mode == "all" or mode == "divs_only":
                datarows[i][i2] *= totdivs
            if mode == "all" or mode == "splits_only":
                datarows[i][i2] *= totsplits
            datarows[i][i2] = round(datarows[i][i2], numdecimals)

    if ascending:
        datarows.reverse()
        datadates.reverse()

    with open(ofile, 'w') as fout:
        for row in datarows:
            s = ""
            for i in range(len(row)):
                s += "{},".format(row[i])
            fout.write("{}\n".format(s[:-1]))

    return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Adjusts historical data for splits/dividends")

    parser.add_argument("datafiles", nargs='*', help="Historical data")

    parser.add_argument("-c", nargs='*', type=int, help="columns to be adjusted")
    parser.add_argument("-s", help="Yahoo Finance symbol")
    parser.add_argument("--sensitivity", type=float, default=1e-1, help="sensitivity setting for automatic column detection")
    parser.add_argument("-f", help="Datetime format for the data file")
    parser.add_argument("-d", type=int, help="Number of decimals")
    parser.add_argument("-m", choices=["all", "divs_only", "splits_only"], help="Adjustment mode")
    parser.add_argument("-o", help="output filename")
    parser.add_argument("-v", action="store_true", help="verbose mode")

    args = parser.parse_args()

    if args.v:
        logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)], level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%j-%H:%M:%S')
    else:
        logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%j-%H:%M:%S')

    for datafile in args.datafiles:


        # first analyze where are the price fields (look for floats of similar amplitude)

        logging.info("Finding price columns from file {}...".format(datafile))

        pricecols, numdecimals = analyze_datafile(datafile, args.sensitivity)

        if args.d:
            numdecimals = args.n

        if args.c:
            pricecols = args.c
            logging.info("Using price columns: {}".format(pricecols))
        else:
            logging.info("Price columns found: {}".format(pricecols))

        logging.info("Number of decimals: {}".format(numdecimals))

        if args.o and len(args.datafiles) == 1:  # only supported with single file
            ofile = args.o
        else:
            fn = os.path.basename(datafile)
            ofile = "{}_adjusted.csv".format(fn[:-4])

        if args.f:
            dtformat = args.f
        else:
            dtformat = get_dt_format(datafile)
            if not dtformat:
                continue
            logging.info("Datetime format found: {}".format(dtformat))

        if args.s and len(args.datafiles) == 1:  # only supported with single file
            yahoosymbol = args.s
        else:
            fn = os.path.basename(datafile)
            m = re.search(".*?[-._]", fn)
            if m:
                yahoosymbol = m.group(0)[:-1]
            else:
                logging.error("Cannot guess Yahoo Finance symbol based on datafile")
                continue

        mindt, maxdt, ascending = analyze_dates(datafile, dtformat)

        if not mindt:  # in case of error
            continue

        # we need one more day of data to accurately construct div/split data
        enddate = maxdt.date() + timedelta(days=1)

        yahoodata = download_yahoo_data(yahoosymbol, mindt.date(), enddate, "d", "x")
        logging.info("Downloaded {} of Yahoo historical data.".format(bytes_to_str(len(yahoodata))))

        logging.info("Compiling dividend/split multipliers based on Yahoo data...")

        dsmult = construct_ds_multipliers(yahoodata)

        if not dsmult:
            logging.info("No dividends/splits, no adjustments necessary.")
            shutil.copy(datafile, ofile)  # no adjustments needed, just copy file as it is
        else:
            logging.info("Doing the necessary adjustments...".format(ofile))
            res = adjust_price_columns(datafile, dsmult, pricecols, ofile, dtformat, numdecimals, args.m)
            if not res:
                continue

        logging.info("Output saved to {}".format(ofile))

