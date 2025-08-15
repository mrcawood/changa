#ifndef PE_LIST_H
#define PE_LIST_H

#ifdef CUDA
#include "ParallelGravity.h"

// Need a separate class for each type of request (particle, node, local, remote)
class PEList : public CBase_PEList
{
    /// TreePieces on this PE
    CkVec<TreePiece*> vtpLocal;
    /// Count of TreePieces with particles on this PE
    NonEmptyTreePieceCounter cTreePieces;

    vector<ILCell> iList;

    vector<int> bucketMarkers;
    int finalBucketMarker;
    vector<int> bucketStarts;
    vector<int> bucketSizes;

    vector<CompactPartData> missedParts;
    vector<CudaMultipoleMoments> missedNodes;

    CudaRequest *request;

    /// Type of request
    int bNode;
    int bRemote;
    int bResume;

    CudaMultipoleMoments *d_localMoments;
    CompactPartData *d_localParts;
    VariablePartData *d_localVars;
    CompactPartData *d_remoteParts;
    CudaMultipoleMoments *d_remoteMoments;
    cudaStream_t stream;
    CkCallback *finishCb;

    cudatype fperiod;

    /// Signal to the TreePieces whether the GPU kernel has completed
    int bGPUDone;
    /// Indicate whether GPU kenel launch was delayed by local data transfer
    int bWaitForLocalData;

  public:
    PEList(int _bNode, int _bRemote, int _bResume) {
        bNode = _bNode;
        bRemote = _bRemote;
        bResume = _bResume;

	finalBucketMarker = -1;
	bWaitForLocalData = 0;
	cudaStreamCreate(&stream);
    }
    PEList(CkMigrateMessage *m) : CBase_PEList(m) {}
    ~PEList() { cudaStreamDestroy(stream); }
    void pup(PUP::er &p) {}

    void finishWalk(TreePiece *treePiece);
    void launchGPUKernel();
    void sendList(TreePiece *treePiece, CudaRequest *data);
    void reset();
    int getNumTPs() { return cTreePieces.count; }
    void setGPUDone(int val) { bGPUDone = val; }
    int isGPUDone() { return bGPUDone; }
    int isWaitingForLocalData() { return bWaitForLocalData; }
};

#endif
#endif
