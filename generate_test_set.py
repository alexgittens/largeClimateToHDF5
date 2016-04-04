# This shoud be exactly the same as the code run for production of the final dataset, just with
# parameters set to run on a smaller dataset for validation that we can map the output back to the input
#
# 50 files 
# 10 files/process
# so need 5 processes
#
# use 5 files in memory at a time
# assume 4 GB/file => 20 GB/process => can fix 6 processes/node => 1 node suffices
#
# use 5 writers
# write 1024 row chunks at a time
# => should write out in 5184 chunks of columns
#
# to run:
# salloc -N 1 -t 30 -p debug --qos=premium
# bash
# module load netcdf4-python h5py python mpi4py
# srun -c 6 -n 5 -u python-mpi -u ./generate_test_set.py T

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

numWriters = 5

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
REDUCEDNUMFILES = 50
filelist = [filelist[idx] for idx in np.arange(REDUCEDNUMFILES)]

numFiles = len(filelist)
filesPerProc = numFiles/numProcs
numFilesInMem = 5
rowTransferChunk = 1024
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
numRowChunksToTransfer = numRows/rowTransferChunk

littlePartitionSize = numRows/numWriters;
bigPartitionSize = littlePartitionSize + 1;
numLittlePartitions = numWriters - numRows % numWriters;
numBigPartitions = numRows % numWriters;

if rank < numBigPartitions:
    myRowsPerOutputFile = bigPartitionSize
else:
    myRowsPerOutputFile = littlePartitionSize

# this should never be violated b/c of our load balanced assumptions
assert(myRowsPerOutputFile == littlePartitionSize), "error"

assert (numFiles % numProcs == 0), "The number of files is not evenly divisible by the number of processes"
assert (filesPerProc % numFilesInMem == 0), "The number of files in memory on each process doesn't evenly divide the number of files the process is responsible for"
assert (numRows % numRowChunksToTransfer == 0), "The number of rows is not divisible by the size of the chunks of rows being transferred" 
assert (numRowChunksToTransfer % numWriters == 0), "The number of row chunks to be transferred is not divisible by the number of writer processes"

report("Starting %d processes" % numProcs)
report("Found %d input files" % len(filelist))
report("Each process is responsible for %d files" % filesPerProc)
report("Each process handles %d files at a time" % numFilesInMem)
report("Transferring %d rows at a time to the writer processes" % rowTransferChunk)
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
reportbarrier("Forcing files to have the right size before starting to write to them")

# just touch the last entry in each file to force them to resize
# this doesn't seem to do anything
if rank in np.arange(numWriters):
    rows[myRowsPerOutputFile - 1, :] = -99999.0
    fout.flush()

reportbarrier("Done setting file sizes")


localcolumncount = numFilesInMem*numtimeslices
rawvar = np.empty((numtimeslices, numlevels, numlats, numlongs), dtype=np.float64)
curvals = np.empty((flattenedlength, localcolumncount), dtype=np.float64)
chunktotransfer = np.empty((rowTransferChunk*localcolumncount,), dtype=np.float64)
if rank in np.arange(numWriters):
    collectedchunk = np.ascontiguousarray(np.empty((numProcs*localcolumncount*rowTransferChunk,), \
            dtype=np.float64))
    chunktowrite = np.ascontiguousarray(np.empty((rowTransferChunk, numProcs*localcolumncount), \
            dtype=np.float64))
else:
    collectedchunk = None

fhoffset = 0
for (varidx,curvar) in enumerate(varnames): 
    reportbarrier("Writing variable %d/%d: %s" % (varidx + 1, numvars, curvar))

    # process the columns in filesPerProc/numFilesInMem chunks
    for colchunkidx in np.arange(filesPerProc/numFilesInMem):
        report("Loading column chunk %d/%d for this variable" % (colchunkidx + 1, filesPerProc/numFilesInMem))

        # fill out the current chunk of columns for this process: 
        # a flattenedlength by numFilesInMem*numtimeslices matrix
        for fhidxOffset in np.arange(numFilesInMem):
            rawvar[:] = myHandles[fhoffset + fhidxOffset][curvar][:]
            startcol = fhidxOffset*numtimeslices
            endcol = (fhidxOffset+1)*numtimeslices
            curvals[:, startcol:endcol] = \
                    np.reshape(rawvar, (flattenedlength, numtimeslices)) # uses C ordering
        fhoffset = fhoffset + numFilesInMem

        reportbarrier("Finished loading column chunk %d/%d for this variable" % \
                (colchunkidx + 1, filesPerProc/numFilesInMem))
        report("Gathering and writing this chunk of columns to file")

        currowoffset = 0
        for rowchunkidx in np.arange(numRowChunksToTransfer):
            # collect rowTransferChunk rows (by numFilesInMem*numtimeslices columns) from each process
            # flatten, and gather at the appropriate writer
            startrow = rowchunkidx * rowTransferChunk
            endrow = (rowchunkidx + 1) * rowTransferChunk
            chunktotransfer[:] = curvals[startrow:endrow, :].flatten()
            MPI.COMM_WORLD.Gather(chunktotransfer, collectedchunk, root = (rowchunkidx % numWriters))

            if rank == (rowchunkidx % numWriters) :
                # reshape the collected chunk to the chunk to be written out, of size
                # rowTransferChunk by numProcs*numFilesInMem*numtimeslices
                for processnum in np.arange(numProcs):
                    startcol = processnum*localcolumncount
                    endcol = (processnum+1)*localcolumncount
                    startidx = processnum*(localcolumncount * rowTransferChunk)
                    endidx = (processnum + 1)*(localcolumncount *rowTransferChunk)
                    chunktowrite[:, startcol:endcol] = np.reshape(collectedchunk[startidx:endidx], \
                            (rowTransferChunk, localcolumncount))

                # write the chunk out
                startcol = colchunkidx * (localcolumncount * numProcs)
                endcol = (colchunkidx + 1) * (localcolumncount * numProcs)
                rows[currowoffset:(currowoffset + rowTransferChunk), startcol:endcol] = chunktowrite
                currowoffset = currowoffset + rowTransferChunk

                if (currowoffset/rowTransferChunk % 2) == 0:
                        status("Wrote row chunk %d/%d of column chunk %d/%d to output file %d/%d" % \
                                (rowchunkidx + 1, numRowChunksToTransfer, \
                                colchunkidx + 1, filesPerProc/numFilesInMem, \
                                rank + 1, numWriters))

reportbarrier("Cleaning up")
for fh in myHandles:
    fh.close()
if rank in np.arange(numWriters):
    fout.close()

