# Algorithm:
# Each processor is responsible for the same number of files, filesPerProc = 10200/numProcs
# To write out one variable, for filesPerProc/numFilesInMem times, repeat:
#  - each processor loads in the variable from the next chunk of files, numFilesInMem, it is responsible for and reshapes it into a flattenedlength-by-8 submatrix to be written into the final matrix
#  - iterating over the flattenedlength rows of these submatrices, extract several rows at a time, writeRowChunkSize, and send those to one of the writer processes in a round robin fashion
#  - each writer process writes out to an independent file
#
# Note, each process will need about 2GB per file, so plan to have numFilesInMem * 2GB at least per processor ( might need more for reshaping, and miscellany)
#
# Assumptions: numProcs evenly divides 10200, numFilesInMem evenly divides filesPerProc, and numWriters evenly divides flattenedlength
#
# Optimizations:
# - turn off fill at allocation in hdf5
# - use an output directory that has 72 OSTs
# - turn on alignment with striping (use output from lfs getstripe to set alignment)
# - set the buffer size
#
# to test speed
# salloc -N 15 -p debug -t 30 --qos=premium
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 6 -n 75 -u python-mpi -u ./concat.py
#
# to run in production
# salloc -N 60 -p debug -t 30 --qos=premium
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 6 -n 300 -u python-mpi -u ./concat.py
#

from mpi4py import MPI
from netCDF4 import Dataset
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import time
import math, sys

rank = MPI.COMM_WORLD.Get_rank()
mpi_info = MPI.Info.Create()
numProcs = MPI.COMM_WORLD.Get_size()

procslist = np.arange(numProcs)

numWriters = min(numProcs, 72) # MIGHT BE SMARTER TO SET MANUALLY

def status(message, ranks=procslist):
    if rank in ranks:
        print "%s, process %d: %s" % (time.asctime(time.localtime()), rank, message)

def report(message):
    status(message, [0])

def reportbarrier(message):
    MPI.COMM_WORLD.Barrier()
    report(message)

datapath = "/global/cscratch1/sd/gittens/large-climate-dataset/data"
filelist = [fname for fname in listdir(datapath) if fname.endswith(".nc")]

# FOR DEBUGGING: REMOVE THIS ON FINAL RUN!
REDUCEDNUMFILES = 5*numProcs
filelist = [filelist[idx] for idx in np.arange(REDUCEDNUMFILES)]

numFiles = len(filelist)
filesPerProc = numFiles/numProcs
numFilesInMem = 5 # assume we want to hold 5 files in memory on each process
numLoadStages = filesPerProc/numFilesInMem

numtimeslices = 8 # each variable comes in 8 time slices, and one time slice becomes one column

#varnames = ["T", "U", "V", "Q", "Z3"]
varnames = ["T"]
numvars = len(varnames)
numlevels = 30
numlats = 768
numlongs = 1152
flattenedlength = numlevels*numlats*numlongs
numRows = flattenedlength*numvars
numCols = len(filelist)*numtimeslices
numRowsPerOutputFile = numRows/numWriters

assert (numFiles % filesPerProc == 0), "The files can't be evenly divided among the processors"
assert (filesPerProc % numFilesInMem == 0), "The number of files in memory on each processor doesn't evenly divide the number of files the processor is responsible for"
assert (flattenedlength % numWriters == 0), "The number of writers doesn't evenly divide the number of rows associated with each variable"

report("Starting %d processes" % numProcs)
report("Found %d input files" % len(filelist))
report("Each process is responsible for %d files" % filesPerProc)
report("Each process handles %d files at a time" % numFilesInMem)
report("This process will require loading data from files in %d stages" % numLoadStages)

myFiles = [fname for (index, fname) in enumerate(filelist) if (index % numProcs == rank)]
myHandles = []
for (idx, fname) in enumerate(myFiles):
    myHandles.append( Dataset(join(datapath, fname), "r") )

# write out the names of the files corresponding to each column in the final output matrix, ordered 0,...,numCols
colfilenames = MPI.COMM_WORLD.gather(myFiles, root=0)
if rank == 0:
   colnames = [item for sublist in colfilenames for item in sublist] 
   print colnames
   with open("colfilenames.txt", "w") as colfout:
       np.save(colfout, np.array(colnames))

reportbarrier("Finished opening all files")
report("Creating file and dataset")

# Open the output files on the writer processes and create the datasets that will contain the rows for that writer
foutname = "superstrided/atmosphere"
if rank in np.arange(numWriters):
    #Ask for alignment with the stripe size (use lfs getstripe on target directory to determine)
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    propfaid.set_alignment(1024, 1024*1024)
    propfaid.set_sieve_buf_size(numCols*8*20) # be able to store 20 rows worth of data
    fid = h5py.h5f.create(join(datapath, foutname + str(rank) + ".hdf5"), flags=h5py.h5f.ACC_TRUNC, fapl=propfaid)
    fout = h5py.File(fid)

    # Don't use filling 
    spaceid = h5py.h5s.create_simple((numRowsPerOutputFile, numCols))
    plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    datasetid = h5py.h5d.create(fout.id, "rows", h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
    rows = h5py.Dataset(datasetid)

reportbarrier("Finished creating file and dataset")

localcolumncount = numFilesInMem*numtimeslices
rawvar = np.empty((numtimeslices, numlevels, numlats, numlongs), dtype=np.float64)
curvals = np.empty((flattenedlength, localcolumncount), dtype=np.float64)
if rank in np.arange(numWriters):
    collectedrowchunk = np.ascontiguousarray(np.empty((numProcs*localcolumncount,), dtype=np.float64))
else:
    collectedrowchunk = None

fhoffset = 0
curoutputrow = 0
for (varidx,curvar) in enumerate(varnames): 
    reportbarrier("Writing variable %d/%d: %s" % (varidx + 1, numvars, curvar))

    # process the columns in filesPerProc/numFilesInMem chunks
    for colchunkidx in np.arange(filesPerProc/numFilesInMem):
        report("Loading column chunk %d/%d for this variable" % (colchunkidx + 1, filesPerProc/numFilesInMem)

        for fhidxOffset in np.arange(numFilesInMem):
            rawvar[:] = myHandles[fhoffset + fhidxOffset][curvar][:]
            startcol = fhindexOffset*numtimeslices
            endcol = (fhindexOffset+1)*numtimeslices
            curvals[:, startcol:endcol] = \
                    np.reshape(rawvar, (flattenedlength, numtimeslices)) # uses C ordering
        fhoffset = fhoffset + 8

        reportbarrier("Finished loading column chunk %d/%d for this variable" % (colchunkidx + 1, filesPerProc/numFilesInMem))
        report("Gathering and writing this chunk of columns to file")

        for rowidx in np.arange(flattenedlength):
            MPI.COMM_WORLD.Gather(curvals[rowoffset, :], collectedrowchunk, root = (rowidx % numWriters))
            if rank in np.arange(numWriters):
                startcol = colchunkidx * (localcolumncount * numProcs)
                endcol = (colchunkidx + 1) * (localcolumncount * numProcs)
                rows[curoutputrow, startcol:endcol] = collectedrowchunk
                curoutputrow = curoutputrow + 1
                if (curoutputrow % 1000) == 0:
                    report("Wrote row %d/%d of column chunk %d/%d to file" % (curoutputrow + 1, numRowsPerOutputFile, colchunkidx + 1, filesPerProc/numFilesInMem))

for fh in enumerate(myHandles):
    fh.close()
if rank in np.arange(numWriters):
    fout.close()

