# Tests that the data has been properly extracted by spot checking a few observations
# Assumes each set of files in the directory contains only one variable
import h5py 
from netCDF4 import Dataset
import numpy as np
from os import listdir
from os.path import join


numtestsamples = 1
varname = 'T'

# should know these from the conversion
numtimeslices = 8 
numlevs = 30
numlats = 768
numlons = 1152
flattenedlength = numlevs * numlats * numlons
rowTransferChunk = 1024 
numProcs = 5
numWriters = 5
numFilesInMem = 5
filesPerProc = 10

rawdatapath = '/global/cscratch1/sd/gittens/large-climate-dataset/data'
datapath ="/global/cscratch1/sd/gittens/large-climate-dataset/data/superstrided"

# load column to file mapping
colmappingname = join(datapath, "colfilenames" + str(varname))
colfnames = np.load(colmappingname)

# sort the output files in the order they were written to, then open them 
filelist = [fname for fname in listdir(datapath) if (fname.startswith(varname) and fname.endswith(".hdf5"))]
filedict = dict([[ fname, int(fname[1:-5]) ] for fname in filelist ]) # assuming the variable name is one character
filelist = sorted(filedict)
outputfhlist = [h5py.File(join(datapath, fname), "r") for fname in filelist]

# map a column to the corresponding file and timeslice
# note that the column in the final output matrix corresponding to the t-th timeslice from the f-th file on the p-th processor on the i-th iteration of writing column chunks
# has index (i-1) * (numFilesInMem * numProcs * numslices) + (p - 1)*(numFilesInMem * numslices) + (f - 1)*numslices + (t-1)
# so just invert this mapping to map columns back to (f, p, i, t)
# then note that the filename this corresponds to has index (p - 1)*filesPerProc + (i - 1)*numFilesInMem + (f-1)
def mapColToOrig(col):
    iteridx = col/(numFilesInMem * numProcs * numtimeslices)
    remainder = col % (numFilesInMem * numProcs * numtimeslices)
    processidx = remainder/(numFilesInMem * numtimeslices)
    remainder = col % (numFilesInMem * numtimeslices)
    fileidx = remainder/numtimeslices
    timeslice = col % numtimeslices

    # figure out which file generated this column
    fnameidx = processidx*filesPerProc + iteridx*numFilesInMem + fileidx
    return (colfnames[fnameidx], timeslice)

# sample columns from the output matrix, and find out the corresponding input file and timeslice of the variable
samplecolindices = np.random.randint(0, numtimeslices*len(colfnames), numtestsamples)
sampleinputfiles = map( lambda colindex: mapColToOrig(colindex)[0], samplecolindices)
sampletimeslices = map( lambda colindex: mapColToOrig(colindex)[1], samplecolindices)

sampletriplets = zip(samplecolindices, sampleinputfiles, sampletimeslices)

# given a sample triplet, compute the difference between the original and converted data

def load_fromfiles(colindex):
    convertedvar = np.empty((flattenedlength,), dtype=np.float64)
    offsetintofile = [0]*numWriters
    for rowchunkidx in np.arange(flattenedlength/rowTransferChunk):
        fileidx = rowchunkidx % numWriters
        convertedvar[rowchunkidx*rowTransferChunk: (rowchunkidx + 1)*rowTransferChunk] = \
                outputfhlist[fileidx].get("rows")[offsetintofile[fileidx]:(offsetintofile[fileidx] + rowTransferChunk), colindex]
        offsetintofile[fileidx] = offsetintofile[fileidx] + rowTransferChunk
    return convertedvar

def test_triplet(triplet):
    (colindex, inputfname, timeslice) = triplet

    # load data from the original file
    inputfh = Dataset(inputfname, "r")
    rawvar = inputfh[varname][timeslice, ...].flatten()
    convertedvar = load_fromfiles(colindex)

    return (rawvar, convertedvar)

(rawvar, convertedvar) = test_triplet(sampletriplets[0])


