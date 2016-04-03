# Algorithm:
# Each processor is responsible for the same number of files, filesPerProc = 10200/numProcs
# To write out one variable, for filesPerProc/numFilesInMem times, repeat:
#  - each processor loads in the variable from the next chunk of files, numFilesInMem, it is responsible for and reshapes it into a flattenedlength-by-8 submatrix to be written into the final matrix
#  - iterating over the flattenedlength rows of these submatrices, extract several rows at a time, writeRowChunkSize, and send those to one of the writer processes in a round robin fashion
#  - each writer process writes out to an independent file
#
# Note, each process will need about 2GB per file, so plan to have numFilesInMem * 5GB at least per processor (since you need more for reshaping, and miscellany)
#
# Assumptions: numProcs evenly divides 10200, numFilesInMem evenly divides filesPerProc, and numWriters evenly divides flattenedlength
#
# Optimizations:
# - turn off fill at allocation in hdf5
# - use an output directory that has 72 OSTs
# - turn on alignment with striping (use output from lfs getstripe to set alignment)
# - set the buffer size
#
# to debug, guesstimate speed, and test for correctness,
# with 1500 files, 10 files per process, 5 files in mem at a time, use
# salloc -N 30 -p debug -t 30 --qos=premium
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 6 -n 150 -u python-mpi -u ./fname.py T
#
# ditto, for 1800 files, 4 files per process, 2 files in mem at time, 
# salloc -N 30 -p debug -t 30 --qos=premium
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 2 -n 450 -u python-mpi -u ./fname.py T
#
# will do conversion one variable at a time
# to run in production: 10200 files, 20 files/process, 2 files in memory at a time, 4GB/file estimated => 34 nodes w/ 510 processes, 2 cores per process
# need an estimate of the time!
# salloc -N 34 -p regular -t 30 --qos=premium
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 2 -n 510 -u python-mpi -u ./fname.py T
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

# MIGHT BE SMARTER TO SET MANUALLY
#numWriters = numProcs
numWriters = 216 # 72 * 3

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
REDUCEDNUMFILES = 510
filelist = [filelist[idx] for idx in np.arange(REDUCEDNUMFILES)]

numFiles = len(filelist)
filesPerProc = numFiles/numProcs
numFilesInMem = 1 
numLoadStages = filesPerProc/numFilesInMem

numtimeslices = 8 # each variable comes in 8 time slices, and one time slice becomes one column

#varnames = ["T", "U", "V", "Q", "Z3"]
varnames = [sys.argv[1]] 
numvars = len(varnames)
numlevels = 30
numlats = 768
numlongs = 1152
flattenedlength = numlevels*numlats*numlongs
numRows = flattenedlength*numvars
numCols = len(filelist)*numtimeslices

littlePartitionSize = numRows/numWriters;
bigPartitionSize = littlePartitionSize + 1;
numLittlePartitions = numWriters - numRows % numWriters;
numBigPartitions = numRows % numWriters;

if rank < numBigPartitions:
    myRowsPerOutputFile = bigPartitionSize
else:
    myRowsPerOutputFile = littlePartitionSize

assert (numFiles % numProcs == 0), "The number of files is not evenly divisible by the number of processes"
assert (filesPerProc % numFilesInMem == 0), "The number of files in memory on each process doesn't evenly divide the number of files the process is responsible for"

report("Starting %d processes" % numProcs)
report("Found %d input files" % len(filelist))
report("Each process is responsible for %d files" % filesPerProc)
report("Each process handles %d files at a time" % numFilesInMem)
report("Using %d writer processes" % numWriters)
report("This process will require loading data from files in %d stages" % numLoadStages)

myFiles = [fname for (index, fname) in enumerate(filelist) if (index % numProcs == rank)]
myHandles = []
for (idx, fname) in enumerate(myFiles):
    myHandles.append( Dataset(join(datapath, fname), "r") )

reportbarrier("Finished opening all files")
report("Creating file and dataset")

# Open the output files on the writer processes and create the datasets that will contain the rows for that writer
foutname = "superstrided/" + str(varnames[0])
coloutname = "superstrided/colfilenames" + str(varnames[0])

# write out the names of the files corresponding to each column in the final output matrix, ordered 0,...,numCols
colfilenames = MPI.COMM_WORLD.gather(myFiles, root=0)
if rank == 0:
   colnames = [item for sublist in colfilenames for item in sublist] 
   with open(coloutname, "w") as colfout:
       np.save(colfout, np.array(colnames))

if rank in np.arange(numWriters):
    #Ask for alignment with the stripe size (use lfs getstripe on target directory to determine)
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    propfaid.set_alignment(1024, 1024*1024)
    propfaid.set_sieve_buf_size(numCols*8*20) # be able to store 20 rows worth of data
    fid = h5py.h5f.create(join(datapath, foutname + str(rank) + ".hdf5"), flags=h5py.h5f.ACC_TRUNC, fapl=propfaid)
    fout = h5py.File(fid)

    # Don't use filling 
    spaceid = h5py.h5s.create_simple((myRowsPerOutputFile, numCols))
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
        report("Loading column chunk %d/%d for this variable" % (colchunkidx + 1, filesPerProc/numFilesInMem))

        for fhidxOffset in np.arange(numFilesInMem):
            rawvar[:] = myHandles[fhoffset + fhidxOffset][curvar][:]
            startcol = fhidxOffset*numtimeslices
            endcol = (fhidxOffset+1)*numtimeslices
            curvals[:, startcol:endcol] = \
                    np.reshape(rawvar, (flattenedlength, numtimeslices)) # uses C ordering
        fhoffset = fhoffset + 8

        reportbarrier("Finished loading column chunk %d/%d for this variable" % (colchunkidx + 1, filesPerProc/numFilesInMem))
        report("Gathering and writing this chunk of columns to file")

        for rowidx in np.arange(flattenedlength):
            MPI.COMM_WORLD.Gather(curvals[rowidx, :], collectedrowchunk, root = (rowidx % numWriters))
            if rank == (rowidx % numWriters) :
                startcol = colchunkidx * (localcolumncount * numProcs)
                endcol = (colchunkidx + 1) * (localcolumncount * numProcs)
                rows[curoutputrow, startcol:endcol] = collectedrowchunk
                if (curoutputrow % 1000) == 0:
                    status("Wrote row %d/%d of column chunk %d/%d to output file %d/%d" % \
                            (curoutputrow + 1, myRowsPerOutputFile, colchunkidx + 1, filesPerProc/numFilesInMem, rank + 1, numWriters))
                curoutputrow = curoutputrow + 1

for fh in enumerate(myHandles):
    fh.close()
if rank in np.arange(numWriters):
    fout.close()

