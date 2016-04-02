from mpi4py import MPI
from h5py import File
import numpy as np

rank = MPI.COMM_WORLD.Get_rank()
numProcs = MPI.COMM_WORLD.Get_size()
procsList = np.arange(numProcs)

def status(message, ranks=procsList):
    if rank in ranks:
        print "%s, process %d/%d: %s" % (time.strftime("%I:%M:%s%p"), rank + 1, numProcs, message)

def report(message):
    status(message, [0])

def reportbarrier(message):
    MPI.COMM_WORLD.Barrier()
    report(message)

fout = File(fname, "w", driver="mpio", comm=MPI.COMM_WORLD)
rows = fout.create_dataset("rows", (numRows, numCols), dtype=np.float64)

