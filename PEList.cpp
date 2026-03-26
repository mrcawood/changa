#ifdef CUDA

#include "PEList.h"
#include "DataManager.h"
#include "HostCUDA.h"

#ifdef TESTSTEP_PELIST_RETRY_DIAG
#define TESTSTEP_RETRY_PRINT(...) CkPrintf("TESTSTEP_PELIST_RETRY_DIAG " __VA_ARGS__)
#else
#define TESTSTEP_RETRY_PRINT(...)
#endif

#ifdef TESTSTEP_LOCALWALK_CHAIN_DIAG
#define TESTSTEP_LOCALWALK_PRINT(...) CmiError("TESTSTEP_LOCALWALK_CHAIN " __VA_ARGS__)
#else
#define TESTSTEP_LOCALWALK_PRINT(...)
#endif

#if defined(TESTSTEP_GPU_PROGRESS_DIAG)
#include "teststep_gpu_progress.h"
#endif

/// @brief Each TreePiece on a given PE checks in as its tree walk completes
///        Once all TreePieces are done, launch a gravity kernel on the GPU
/// @param treePiece A reference to the TreePiece that checked in
void PEList::finishWalk(TreePiece *treePiece) {
    vtpLocal.push_back(treePiece);
    TESTSTEP_LOCALWALK_PRINT(
        "ev=peList_finishWalk_enter pe=%d node=%d pelist=%p tp=%p gathered=%d expected=%d bNode=%d bRemote=%d bResume=%d delayed=%d finishCb=%p\n",
        CkMyPe(), CkMyNode(), (void *)this, (void *)treePiece, (int)vtpLocal.size(),
        cTreePieces.count, bNode, bRemote, bResume, bKernelDelayed, (void *)finishCb);

    // On first call, find the total number of active pieces on this PE.
    // The charm++ location manager gives us this count in cTreePieces
    if(vtpLocal.size() == 1) {
        CkLocMgr *locMgr = treeProxy.ckLocMgr();
        locMgr->iterate(cTreePieces);
    }

    // check if we have everyone
    if(vtpLocal.size() < cTreePieces.count)
        return;

    // bucketMarkers[i+1] is needed to determine # of IL entries per bucket
    if(finalBucketMarker != -1)
	bucketMarkers.push_back(finalBucketMarker);

    finishCb = new CkCallback(CkIndex_PEList::finishWalkCb(), CkMyPe(), thisProxy);
    TESTSTEP_LOCALWALK_PRINT(
        "ev=peList_finishWalk_registerCb pe=%d node=%d pelist=%p finishCb=%p gathered=%d expected=%d\n",
        CkMyPe(), CkMyNode(), (void *)this, (void *)finishCb, (int)vtpLocal.size(),
        cTreePieces.count);

    // If the DataManager device pointer is NULL, the GPU data transfer is
    // still in progress and we need to delay the kernel launch.
    // launchKernel() always uses d_localMoments/d_localParts; remote also needs remote data.
    DataManager *dm = dMProxy.ckLocalBranch();
    bool dataReady = dm->bLocalDataTransferred.load() &&
        (!bRemote || dm->bRemoteDataTransferred.load());
    TESTSTEP_RETRY_PRINT(
        "finishWalk pe=%d node=%d this=%p bNode=%d bRemote=%d bResume=%d delayedBefore=%d localReady=%d remoteReady=%d dataReady=%d action=%s\n",
        CkMyPe(), CkMyNode(), (void*)this, bNode, bRemote, bResume, bKernelDelayed,
        dm->bLocalDataTransferred.load(), dm->bRemoteDataTransferred.load(),
        dataReady, dataReady ? "launch" : "setDelayed");
    if (!dataReady) {
        bKernelDelayed = 1;
        TESTSTEP_LOCALWALK_PRINT(
            "ev=peList_finishWalk_decision pe=%d node=%d pelist=%p action=setDelayed delayed=%d localReady=%d remoteReady=%d gathered=%d expected=%d\n",
            CkMyPe(), CkMyNode(), (void *)this, bKernelDelayed,
            (int)dm->bLocalDataTransferred.load(), (int)dm->bRemoteDataTransferred.load(),
            (int)vtpLocal.size(), cTreePieces.count);
        TESTSTEP_RETRY_PRINT(
            "finishWalk pe=%d node=%d this=%p delayedAfter=%d\n",
            CkMyPe(), CkMyNode(), (void*)this, bKernelDelayed);
    } else {
        TESTSTEP_LOCALWALK_PRINT(
            "ev=peList_finishWalk_decision pe=%d node=%d pelist=%p action=launch delayed=%d localReady=%d remoteReady=%d gathered=%d expected=%d\n",
            CkMyPe(), CkMyNode(), (void *)this, bKernelDelayed,
            (int)dm->bLocalDataTransferred.load(), (int)dm->bRemoteDataTransferred.load(),
            (int)vtpLocal.size(), cTreePieces.count);
        launchKernel();
    }
}

/// @brief Called from DataManager when remote or local transfer finishes.
/// Launch our kernel only if it was delayed AND the data we need is now ready.
/// launchKernel() always uses d_localMoments/d_localParts; remote also needs remote data.
void PEList::tryLaunchDelayedKernel() {
    TESTSTEP_LOCALWALK_PRINT(
        "ev=peList_tryLaunch_enter pe=%d node=%d pelist=%p delayed=%d bNode=%d bRemote=%d bResume=%d finishCb=%p\n",
        CkMyPe(), CkMyNode(), (void *)this, bKernelDelayed, bNode, bRemote, bResume,
        (void *)finishCb);
    if (!bKernelDelayed) {
        DataManager *dmEarly = dMProxy.ckLocalBranch();
        bool dataReadyEarly = dmEarly->bLocalDataTransferred.load() &&
            (!bRemote || dmEarly->bRemoteDataTransferred.load());
        TESTSTEP_RETRY_PRINT(
            "tryLaunch pe=%d node=%d this=%p bNode=%d bRemote=%d bResume=%d delayed=%d localReady=%d remoteReady=%d dataReady=%d decision=return\n",
            CkMyPe(), CkMyNode(), (void*)this, bNode, bRemote, bResume, bKernelDelayed,
            dmEarly->bLocalDataTransferred.load(), dmEarly->bRemoteDataTransferred.load(), dataReadyEarly);
        return;
    }
    DataManager *dm = dMProxy.ckLocalBranch();
    bool dataReady = dm->bLocalDataTransferred.load() &&
        (!bRemote || dm->bRemoteDataTransferred.load());
    TESTSTEP_RETRY_PRINT(
        "tryLaunch pe=%d node=%d this=%p bNode=%d bRemote=%d bResume=%d delayed=%d localReady=%d remoteReady=%d dataReady=%d decision=%s\n",
        CkMyPe(), CkMyNode(), (void*)this, bNode, bRemote, bResume, bKernelDelayed,
        dm->bLocalDataTransferred.load(), dm->bRemoteDataTransferred.load(),
        dataReady, dataReady ? "launch" : "return");
    if (!dataReady) {
        TESTSTEP_LOCALWALK_PRINT(
            "ev=peList_tryLaunch_decision pe=%d node=%d pelist=%p action=return_not_ready delayed=%d localReady=%d remoteReady=%d\n",
            CkMyPe(), CkMyNode(), (void *)this, bKernelDelayed,
            (int)dm->bLocalDataTransferred.load(), (int)dm->bRemoteDataTransferred.load());
        return;
    }
    TESTSTEP_RETRY_PRINT(
        "tryLaunch_callLaunch pe=%d node=%d this=%p bRemote=%d localReady=%d remoteReady=%d ptrs=(%p,%p,%p)\n",
        CkMyPe(), CkMyNode(), (void*)this, bRemote,
        dm->bLocalDataTransferred.load(), dm->bRemoteDataTransferred.load(),
        (void*)dm->d_localMoments, (void*)dm->d_localParts, (void*)dm->d_localVars);
    bKernelDelayed = 0;
    TESTSTEP_LOCALWALK_PRINT(
        "ev=peList_tryLaunch_decision pe=%d node=%d pelist=%p action=launch delayed=%d localReady=%d remoteReady=%d\n",
        CkMyPe(), CkMyNode(), (void *)this, bKernelDelayed,
        (int)dm->bLocalDataTransferred.load(), (int)dm->bRemoteDataTransferred.load());
    launchKernel();
}

void PEList::finishWalkCb() {
     TESTSTEP_LOCALWALK_PRINT(
         "ev=peList_finishWalkCb_enter pe=%d node=%d pelist=%p finishCb=%p req=%p delayed=%d bNode=%d bRemote=%d bResume=%d\n",
         CkMyPe(), CkMyNode(), (void *)this, (void *)finishCb, (void *)request,
         bKernelDelayed, bNode, bRemote, bResume);
#if defined(TESTSTEP_GPU_PROGRESS_DIAG)
     gpu_prog_cb_fired(GP_PELIST);
#endif
     dMProxy.ckLocalBranch()->transferParticleVarsBack();
     reset();
}

/// @brief Launch the corresponding CUDA kernel, depending what type of request this was
void PEList::launchKernel() {
    TESTSTEP_LOCALWALK_PRINT(
        "ev=peList_launchKernel_enter pe=%d node=%d pelist=%p finishCb=%p delayed=%d bNode=%d bRemote=%d bResume=%d ilist=%d buckets=%d\n",
        CkMyPe(), CkMyNode(), (void *)this, (void *)finishCb, bKernelDelayed,
        bNode, bRemote, bResume, (int)iList.size(), (int)bucketSizes.size());
    request = new CudaRequest;

    // Ensure required data is present on the GPU
    CkAssert(dMProxy.ckLocalBranch()->d_localMoments != nullptr);
    CkAssert(dMProxy.ckLocalBranch()->d_localParts != nullptr);
    CkAssert(dMProxy.ckLocalBranch()->d_localVars != nullptr);
    // The following checks can fail if remote prefetch does not bring in any data.
    // if (bRemote) {
    //    CkAssert(dMProxy.ckLocalBranch()->d_remoteParts != nullptr);
    //    CkAssert(dMProxy.ckLocalBranch()->d_remoteMoments != nullptr);
    // }

    request->d_localMoments = dMProxy.ckLocalBranch()->d_localMoments;
    request->d_localParts = dMProxy.ckLocalBranch()->d_localParts;
    request->d_localVars = dMProxy.ckLocalBranch()->d_localVars;
    request->d_remoteParts = dMProxy.ckLocalBranch()->d_remoteParts;
    request->d_remoteMoments = dMProxy.ckLocalBranch()->d_remoteMoments;
    request->stream = stream;

    request->numBucketsPlusOne = bucketSizes.size()+1;

    request->node = bNode;
    request->remote = bRemote;

    request->fperiod = fperiod;

    const char* funcTag = "PEList::finish";
    if (iList.size() > 0) {
      hostMalloc((void**)&iListHost, iList.size()*sizeof(ILCell), funcTag);
      memcpy(iListHost, iList.data(),  iList.size()*sizeof(ILCell));
      hostMalloc((void**)&bucketMarkersHost, bucketMarkers.size()*sizeof(int), funcTag);
      memcpy(bucketMarkersHost, bucketMarkers.data(), bucketMarkers.size()*sizeof(int));
      hostMalloc((void**)&bucketStartsHost, bucketStarts.size()*sizeof(int), funcTag);
      memcpy(bucketStartsHost, bucketStarts.data(), bucketStarts.size()*sizeof(int));
      hostMalloc((void**)&bucketSizesHost, bucketSizes.size()*sizeof(int), funcTag);
      memcpy(bucketSizesHost, bucketSizes.data(), bucketSizes.size()*sizeof(int));
    }

    if (missedParts.size() > 0) {
      hostMalloc((void**)&missedPartsHost, missedParts.size()*sizeof(CompactPartData), funcTag);
      memcpy(missedPartsHost, missedParts.data(), missedParts.size()*sizeof(CompactPartData));
    }
    if (missedNodes.size() > 0) {
      hostMalloc((void**)&missedNodesHost, missedNodes.size()*sizeof(CudaMultipoleMoments), funcTag);
      memcpy(missedNodesHost, missedNodes.data(), missedNodes.size()*sizeof(CudaMultipoleMoments));
    }

    request->list = iListHost;
    request->missedParts = missedPartsHost;
    request->missedNodes = missedNodesHost;
    request->sMissed = bNode ? missedNodes.size()*sizeof(CudaMultipoleMoments) : missedParts.size()*sizeof(CompactPartData);
    request->bucketMarkers = bucketMarkersHost;
    request->bucketStarts = bucketStartsHost;
    request->bucketSizes = bucketSizesHost;
    request->numInteractions = iList.size();

    request->cb = finishCb;
    TESTSTEP_LOCALWALK_PRINT(
        "ev=peList_launchKernel_submit pe=%d node=%d pelist=%p req=%p cb=%p interactions=%d numBuckets=%d missedParts=%d missedNodes=%d\n",
        CkMyPe(), CkMyNode(), (void *)this, (void *)request, (void *)request->cb,
        request->numInteractions, request->numBucketsPlusOne - 1, (int)missedParts.size(),
        (int)missedNodes.size());

    void (*transferFunc)(CudaRequest*);
    if (bNode) {
        transferFunc = bRemote ? PEListNodeListDataTransferRemote : PEListNodeListDataTransferLocal;
        if (bResume) {
                transferFunc = PEListNodeListDataTransferRemoteResume;
        }
    } else {
        transferFunc = bRemote ? PEListPartListDataTransferRemote : PEListPartListDataTransferLocal;
        if (bResume) {
                transferFunc = PEListPartListDataTransferRemoteResume;
        }
    }

    transferFunc(request);
}

/// @brief Collect the interaction list results each a Compute operation completes
/// @param treePiece The TreePiece that sent the operation
/// @data A CudaRequest object containing the interaction list data
void PEList::sendList(TreePiece *treePiece, CudaRequest* data) {
    int numBucketsPlusOne = data->numBucketsPlusOne;
    int numBuckets = numBucketsPlusOne-1;

    // bucketMarkers need an offset because we are concatenating the interaction lists
    for (int i = 0; i < numBuckets; i++) {
	bucketMarkers.push_back(data->bucketMarkers[i] + iList.size());
    }
    finalBucketMarker = data->bucketMarkers[numBuckets] + iList.size();

    for (int i = 0; i < numBuckets; i++) {
	bucketStarts.push_back(data->bucketStarts[i]);
	bucketSizes.push_back(data->bucketSizes[i]);
    }

    // If we have missed parts/nodes, the indices in the interaction list
    // need to be shifted because the remote data is being concatenated
    if (data->missedParts) {
	// Note that many TreePieces will have the same missed particles
	// We are copying a lot of duplicate data to the GPU here
	int numMissedParts = data->sMissed/sizeof(CompactPartData);
	int missedOffset = missedParts.size();
	for (int i = 0; i < data->numInteractions; i++) {
	    ((ILCell *)data->list)[i].index += missedOffset;
	    iList.push_back(((ILCell *)data->list)[i]);
	}
	for (int i = 0; i < numMissedParts; i++) {
	    missedParts.push_back(((CompactPartData *)data->missedParts)[i]);
	}
    } else if (data->missedNodes) {
	int numMissedNodes = data->sMissed/sizeof(CudaMultipoleMoments);
	int missedOffset = missedNodes.size();
	for (int i = 0; i < data->numInteractions; i++) {
	    ((ILCell *)data->list)[i].index += missedOffset;
	    iList.push_back(((ILCell *)data->list)[i]);
	}
	for (int i = 0; i < numMissedNodes; i++) {
	    missedNodes.push_back(((CudaMultipoleMoments *)data->missedNodes)[i]);
	}
    } else {
	for (int i = 0; i < data->numInteractions; i++) {
	    iList.push_back(((ILCell *)data->list)[i]);
	}
    }

    fperiod = data->fperiod;

    // Call finishBucket for all buckets involved in this interaction
    treePiece->cudaFinishAffectedBuckets(data->affectedBuckets, numBuckets, bRemote);

    // deallocate the memory used by the incoming cudaRequest
    free(data->list);
    free(data->bucketMarkers);
    free(data->bucketStarts);
    free(data->bucketSizes);
    delete[] data->affectedBuckets;
}

/// @brief Re-initalize data arrays and clean up callback objects at the end of the step
void PEList::reset() {
    const char* funcTag = "PEList::reset";
    if (iList.size() > 0) {
      hostFree(iListHost, funcTag);
      hostFree(bucketMarkersHost, funcTag);
      hostFree(bucketStartsHost, funcTag);
      hostFree(bucketSizesHost, funcTag);
    }
    if (missedParts.size() > 0) {
      hostFree(missedPartsHost, funcTag);
    }
    if (missedNodes.size() > 0) {
      hostFree(missedNodesHost, funcTag);
    }
    iList.clear();
    missedParts.clear();
    missedNodes.clear();
    bucketMarkers.clear();
    bucketStarts.clear();
    bucketSizes.clear();

    cTreePieces.reset();
    vtpLocal.clear();
    bRemoteReady = 0;
    bKernelDelayed = 0;
    TESTSTEP_RETRY_PRINT(
        "reset pe=%d node=%d this=%p bNode=%d bRemote=%d bResume=%d delayedAfterReset=%d\n",
        CkMyPe(), CkMyNode(), (void*)this, bNode, bRemote, bResume, bKernelDelayed);
    finalBucketMarker = -1;
    delete finishCb;
    delete request;
    request = nullptr;
}

#endif
