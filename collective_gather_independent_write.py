# to test speed on 10 nodes
# salloc -N 10 --reservation=m1523 -t 30
# bash
# module load h5py-parallel mpi4py netcdf4-python
# srun -c 6 -n 50 -u python-mpi -u ./fname.py variablename
#
# Optimizations:
# - turn off fill at allocation in hdf5
# - use an output directory that has 140 OSTs
# - turn on alignment with striping (use output from lfs getstripe to set alignment)
#
# Note: apparently, the number of aggregator processes is set to the number of OSTs

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

numProcessesPerNode = 5

procslist = np.arange(numProcs)

def status(message, ranks=procslist):
    if rank in ranks:
        print "%s, process %d: %s" % (time.asctime(time.localtime()), rank, message)

def report(message):
    status(message, [0])

def reportbarrier(message):
    MPI.COMM_WORLD.Barrier()
    report(message)

# maps the chunkidx (0...numlevdivs=numWriters) to the rank of the process that should write it out
def chunkidxToWriter(chunkidx):
   return chunkidx*numProcessesPerNode%numProcs # should always write from different nodes
jialinpath =  "/global/cscratch1/sd/jialin/climate/large"
datapath = "/global/cscratch1/sd/gittens/large-climate-dataset/data"
filelist = [fname for fname in listdir(datapath) if fname.endswith(".nc")]
varname = (sys.argv[1])

# FOR TESTING ONLY: REMOVE WHEN RUNNING FINAL JOB
#filelist = [fname for fname in filelist[:1000]]

report("Using %d processes" % numProcs)
report("Writing variable %s" % varname)
report("Found %d input files, starting to open" % len(filelist))

assert( len(filelist) % numProcs == 0)

# open all the files associated with this process and keep handles around (avoid metadata costs)
myfiles = [fname for (index, fname) in enumerate(filelist) if (index % numProcs == rank)]
numFilesPerProc = len(myfiles)
myhandles = [None]*len(myfiles)
for (idx, fname) in enumerate(myfiles):
    myhandles[idx] = Dataset(join(datapath, fname), "r") 

reportbarrier("Finished opening all files")

#varnames = ["T", "U", "V", "Q", "Z3"]
#varnames = ["T"]
varnames = [varname]
numvars = len(varnames)
numvars = 1
numtimeslices = 8
numlevels = 15
#numlevels = 30
numlats = 768
numlongs = 1152
numlevdivs = 64
flattenedlength = numlevels*numlats*numlongs
numRows = flattenedlength*numvars
numCols = len(filelist)*numtimeslices
rowChunkSize = numlats*numlongs/numlevdivs

numWriters = numlevdivs 
coloutname = "superstrided/colfilenames" 
foutname = "superstrided/" + varname
assert ((numlats * numlongs) % numlevdivs == 0)

report("Creating files and datasets")

# Write out the names of the files corresponding to each column in the final output matrix, ordered 0,...,numCols
# simplify!
colfilenames = MPI.COMM_WORLD.gather(myfiles, root=0)
if rank == 0:
   colnames = [item for sublist in colfilenames for item in sublist] 
   with open(coloutname, "w") as colfout:
       np.save(colfout, np.array(colnames))

#Open the output files on the writer processes and create the datasets that will contain the rows for that writer
for chunkidx in np.arange(numWriters):
    if rank == chunkidxToWriter(chunkidx):
        #Ask for alignment with the stripe size (use lfs getstripe on target directory to determine)
        propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        propfaid.set_alignment(1024, 1024*1024)
        #propfaid.set_sieve_buf_size(numCols*8*20) # be able to store 20 rows worth of data
        fid = h5py.h5f.create(join(jialinpath, foutname + str(chunkidx) + ".hdf5"), flags=h5py.h5f.ACC_TRUNC, fapl=propfaid)
        fout = h5py.File(fid)

        # Don't use filling 
        spaceid = h5py.h5s.create_simple((numvars*numlevels*rowChunkSize, numCols))
        plist = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        plist.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
        datasetid = h5py.h5d.create(fout.id, "rows", h5py.h5t.NATIVE_DOUBLE, spaceid, plist)
        rows = h5py.Dataset(datasetid)

reportbarrier("Finished creating files and datasets")

localcolumncount = numFilesPerProc*numtimeslices
curlevdata = np.empty((numlats*numlongs, localcolumncount), \
        dtype=np.float32)
chunktotransfer = np.empty((rowChunkSize*localcolumncount,), dtype=np.float32)

if rank in map(chunkidxToWriter, np.arange(numWriters)):
    collectedchunk = np.ascontiguousarray(np.empty((numCols*rowChunkSize,), \
            dtype=np.float32))
    chunktowrite = np.ascontiguousarray(np.empty((rowChunkSize, numCols), \
            dtype=np.float32))
else:
    collectedchunk = None
curlevdatatemp=np.ascontiguousarray(np.zeros((numlats*numlongs*numtimeslices), \
            dtype=np.float32))
currowoffset = 0
for (varidx,curvar) in enumerate(varnames): 
    reportbarrier("Writing variable %d/%d: %s" % (varidx + 1, numvars, curvar))

    for curlev in np.arange(numlevels):

        # load the data for this level from my files
        reportbarrier("Loading data for level %d/%d" % (curlev + 1, numlevels))
        for (fhidx, fh) in enumerate(myhandles):
            if fh[curvar].shape[0] < numtimeslices and fh[curvar].shape[0] >0:
                status("File %s has only %d timesteps for variable %s, simply repeating the first timestep" % (myfiles[fhidx], fh[curvar].shape[0], curvar))
                for idx in np.arange(numtimeslices):
                    curlevdatatemp[numlats*numlongs*idx:numlats*numlongs*(idx+1)] = fh[curvar][0, curlev, ...].flatten()
		curlevdata[:, fhidx*numtimeslices: (fhidx + 1)*numtimeslices]=curlevdatatemp.reshape(numlats*numlongs, numtimeslices)
            elif fh[curvar].shape[0] ==0:
		status("File %s has only %d timesteps for variable %s, simply repeating the first timestep" % (myfiles[fhidx], fh[curvar].shape[0], curvar))
		curlevdata[:, fhidx*numtimeslices: (fhidx + 1)*numtimeslices] = \
			curlevdatatemp.reshape(numlats*numlongs, numtimeslices)
	    else:
                curlevdata[:, fhidx*numtimeslices: (fhidx + 1)*numtimeslices] = \
                    fh[curvar][:, curlev, ...].reshape(numlats*numlongs, numtimeslices)
        reportbarrier("Done loading data for this level")
        
        # write out this level in several chunks of rows
        reportbarrier("Gathering data for this level from processes to writers")
        for chunkidx in np.arange(numlevdivs):
            startrow = chunkidx*rowChunkSize
            endrow = startrow + rowChunkSize
            chunktotransfer[:] = curlevdata[startrow:endrow, :].flatten()
            MPI.COMM_WORLD.Gather(chunktotransfer, collectedchunk, root = chunkidxToWriter(chunkidx))
        reportbarrier("Done gathering")

        reportbarrier("Writing data for this level on writers")
        for chunkidx in np.arange(numlevdivs):
            if rank == chunkidxToWriter(chunkidx):
                # reshape the collected chunk to the chunk to be written out, of size
                # rowChunkSize by numCols
		print "current rank writer is:%d"%rank
                for processnum in np.arange(numProcs):
                    startcol = processnum*localcolumncount
                    endcol = (processnum+1)*localcolumncount
                    startidx = processnum*(localcolumncount * rowChunkSize)
                    endidx = (processnum + 1)*(localcolumncount *rowChunkSize)
                    chunktowrite[:, startcol:endcol] = np.reshape(collectedchunk[startidx:endidx], \
                            (rowChunkSize, localcolumncount))
		#chunktowrite=np.reshape(collectedchunk,(rowChunkSize,numCols))
                rows[currowoffset : (currowoffset + rowChunkSize)] = chunktowrite
                currowoffset = currowoffset + rowChunkSize
        reportbarrier("Done writing")

for fh in myhandles:
    fh.close()
fout.close()

