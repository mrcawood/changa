#ifndef __CACHEMANAGER_H__
#define __CACHEMANAGER_H__
#include <sys/types.h>
#include <vector>
#include <map>
#include <set>
#include "SFC.h"
#include "TreeNode.h"
#include "GenericTreeNode.h"
#include "charm++.h"

#if COSMO_STATS > 0
#include <fstream>
#endif

/** NodeCacheEntry represents the entry for a remote 
node that is requested by the chares 
on a processor.
It stores the index of the remote chare from 
which node is to be requested and the local
chares that request it.***/

using namespace std;
using namespace TreeStuff;
//using namespace SFC;

typedef GenericTreeNode CacheNode;
typedef u_int64_t CacheKey;

class FillNodeMsg;
class FillParticleMsg;

#include "CacheManager.decl.h"

class CacheStatistics {
  u_int64_t nodesArrived;
  u_int64_t nodesMessages;
  u_int64_t nodesDuplicated;
  u_int64_t nodesMisses;
  u_int64_t nodesLocal;
  u_int64_t particlesArrived;
  u_int64_t particlesTotalArrived;
  u_int64_t particlesMisses;
  u_int64_t particlesLocal;
  u_int64_t particlesError;
  u_int64_t totalNodesRequested;
  u_int64_t totalParticlesRequested;
  u_int64_t maxNodes;
  u_int64_t maxParticles;
  u_int64_t nodesNotUsed;
  int index;

  CacheStatistics() : nodesArrived(0), nodesMessages(0), nodesDuplicated(0), nodesMisses(0),
    nodesLocal(0), particlesArrived(0), particlesTotalArrived(0),
    particlesMisses(0), particlesLocal(0), particlesError(0), totalNodesRequested(0),
    totalParticlesRequested(0), maxNodes(0), maxParticles(0), nodesNotUsed(0), index(-1) { }

 public:
  CacheStatistics(u_int64_t na, u_int64_t nmsg, u_int64_t nd, u_int64_t nm,
		  u_int64_t nl, u_int64_t pa, u_int64_t pta, u_int64_t pm,
		  u_int64_t pl, u_int64_t pe, u_int64_t tnr, u_int64_t tpr,
		  u_int64_t mn, u_int64_t mp, u_int64_t nnu, int i) :
    nodesArrived(na), nodesMessages(nmsg), nodesDuplicated(nd), nodesMisses(nm),
    nodesLocal(nl), particlesArrived(pa), particlesTotalArrived(pta), particlesMisses(pm),
    particlesLocal(pl), particlesError(pe), totalNodesRequested(tnr),
    totalParticlesRequested(tpr), maxNodes(mn), maxParticles(mp), nodesNotUsed(nnu), index(i) { }

  void printTo(CkOStream &os) {
    os << "  Cache: " << nodesArrived << " nodes (of which " << nodesDuplicated;
    os << " duplicated) arrived in " << nodesMessages;
    os << " messages, " << nodesLocal << " from local TreePieces" << endl;
    os << "  Cache: " << particlesTotalArrived << " particles arrived (corresponding to ";
    os << particlesArrived << " remote nodes), " << particlesLocal << " from local TreePieces" << endl;
    if (particlesError > 0) {
      os << "Cache: ======>>>> ERROR: " << particlesError << " particles arrived without being requested!! <<<<======" << endl;
    }
    os << "  Cache: " << nodesMisses << " nodes and " << particlesMisses << " particle misses during computation, " << nodesNotUsed << " never used" << endl;
    os << "  Cache: Maximum of " << maxNodes << " nodes and " << maxParticles << " particles stored at a time in processor " << index << endl;
    os << "  Cache: local TreePieces requested " << totalNodesRequested << " nodes and ";
    os << totalParticlesRequested << " particle buckets" << endl;
  }

  static CkReduction::reducerType sum;

  static CkReductionMsg *sumFn(int nMsg, CkReductionMsg **msgs) {
    CacheStatistics ret;
    ret.maxNodes = 0;
    ret.maxParticles = 0;
    for (int i=0; i<nMsg; ++i) {
      CkAssert(msgs[i]->getSize() == sizeof(CacheStatistics));
      CacheStatistics *data = (CacheStatistics *)msgs[i]->getData();
      ret.nodesArrived += data->nodesArrived;
      ret.nodesMessages += data->nodesMessages;
      ret.nodesDuplicated += data->nodesDuplicated;
      ret.nodesMisses += data->nodesMisses;
      ret.nodesLocal += data->nodesLocal;
      ret.particlesArrived += data->particlesArrived;
      ret.particlesTotalArrived += data->particlesTotalArrived;
      ret.particlesMisses += data->particlesMisses;
      ret.particlesLocal += data->particlesLocal;
      ret.totalNodesRequested += data->totalNodesRequested;
      ret.totalParticlesRequested += data->totalParticlesRequested;
      ret.nodesNotUsed += data->nodesNotUsed;
      if (data->maxNodes+data->maxParticles > ret.maxNodes+ret.maxParticles) {
	ret.maxNodes = data->maxNodes;
	ret.maxParticles = data->maxParticles;
	ret.index = data->index;
      }
    }
    return CkReductionMsg::buildNew(sizeof(CacheStatistics), &ret);
  }
};

class RequestorData {//: public CkPool<RequestorData, 128> {
 public:
  int arrayID;
  int reqID;
  bool isPrefetch;

  RequestorData(int a, int r, bool ip) {
    arrayID = a;
    reqID = r;
    isPrefetch = ip;
  }
};

class CacheEntry {
public:
	CacheKey requestID; // node or particle ID
	int home; // index of the array element that contains this node
	vector<RequestorData> requestorVec;
	//vector<int> requestorVec; // index of the array element that made the request
	//vector<BucketGravityRequest *> reqVec;  //request data structure that is different for each requestor

	bool requestSent;
	bool replyRecvd;
#if COSMO_STATS > 1
	/// total number of requests to this cache entry
	int totalRequests;
	/// total number of requests that missed this entry, if the request is
	/// to another TreePiece in the local processor we never miss
	int misses;
#endif
	CacheEntry(){
		replyRecvd = false;
		requestSent=false;
		home = -1;
#if COSMO_STATS > 1
		totalRequests=0;
		misses=0;
#endif
	}

};

class NodeCacheEntry : public CacheEntry, public CkPool<NodeCacheEntry, 64> {
public:
	CacheNode *node;
	
	NodeCacheEntry():CacheEntry(){
		node = NULL;
	}
	~NodeCacheEntry(){
		CkAssert(requestorVec.empty());
		delete node;
		//reqVec.clear();
	}
	/// 
	void sendRequest(int);
};	

class ParticleCacheEntry : public CacheEntry, public CkPool<ParticleCacheEntry, 64> {
public:
	ExternalGravityParticle *part;
	//int num;
	int begin;
	int end;
	ParticleCacheEntry():CacheEntry(){
		part = NULL;
		begin=0;
		end=0;
	}
	~ParticleCacheEntry(){
		CkAssert(requestorVec.empty());
		delete []part;
		//reqVec.clear();
	}
	void sendRequest(int);
};

class MapKey {//: public CkPool<MapKey, 64> {
public:
	CacheKey k;
	int home;
	MapKey(){
		k=0;
		home=0;
	};
	MapKey(CacheKey _k,int _home){
		k = _k;
		home = _home;
	}
};
bool operator<(MapKey lhs,MapKey rhs);

class CommData {
 public:
  int nodes;
  int particles;
  CommData() : nodes(0), particles(0) {}
};

class CacheManager : public CBase_CacheManager {
private:

  /***********************************************************************
   * Variables definitions
   ***********************************************************************/

  /// Number of chunks in which the tree is splitted
  int numChunks;
  //int newChunks; //<Number of chunks for the next iteration
  /// Nodes currently used as roots for remote computation
  Tree::NodeKey *prefetchRoots;

	/* The Cache Table is fully associative 
	A hashtable can be used as well.*/

	map<CacheKey,NodeCacheEntry *> *nodeCacheTable;
#if COSMO_STATS > 0
	/// nodes arrived from remote processors
	u_int64_t nodesArrived;
	/// messages arrived from remote processors filling nodes
	u_int64_t nodesMessages;
	/// nodes that have arrived more than once from remote processors, they
	/// are counted also as nodesArrived
	u_int64_t nodesDuplicated;
	/// nodes missed while walking the tree for computation
	u_int64_t nodesMisses;
	/// nodes that have been imported from local TreePieces
	u_int64_t nodesLocal;
	// nodes that have been prefetched, but never used during the
	// computation (nodesDuplicated are not included in this count)
	//u_int64_t nodesNeverUsed;
	/// particles arrived from remote processors, this counts only the entries in the cache
	u_int64_t particlesArrived;
	/// particles arrived from remote processors, this counts the real
	/// number of particles arrived
	u_int64_t particlesTotalArrived;
	/// particles missed while walking the tree for computation
	u_int64_t particlesMisses;
	/// particles that have been imported from local TreePieces
	u_int64_t particlesLocal;
	/// particles arrived which were never requested, basically errors
	u_int64_t particlesError;
	/** counts the total number of nodes requested by all
	the chares on the processor***/
	u_int64_t totalNodesRequested;
	/** counts the total number of particles requested by all
	the chares on the processor***/
	u_int64_t totalParticlesRequested;
	/// maximum number of nodes stored at some point in the cache
	u_int64_t maxNodes;
	/// maximum number of nodes stored at some point in the cache
	u_int64_t maxParticles;
	/// nodes that have been fetched but never used in the tree walk
	u_int64_t nodesNotUsed;
#endif
#if COSMO_DEBUG > 0
	ofstream *ofs;
#endif

	/// used to generate new Nodes of the correct type (inheriting classes of CacheNode)
	CacheNode *prototype;
	/// list of TreePieces registered to this branch together with their roots
	map<int,GenericTreeNode*> registeredChares;
	map<int,int> localIndicesMap;

        CkCallback callback;
        CkLocMgr *treePieceLocMgr;
        LBDatabase *lbdb;
        const LDOMHandle *omhandle;
        int localIndices[64];
        CommData *sentData;

#ifdef CACHE_TREE
	/// root of the super-tree hold in this processor
	GenericTreeNode *root;
	/// lookup table for the super-tree nodes
	NodeLookupType nodeLookupTable;
	/// Lookup table for the chunkRoots
	NodeLookupType chunkRootTable;
#endif

	/// weights of the chunks in which the tree is divided, the cache will
	/// update the chunk division based on these values
	u_int64_t *chunkWeight;

	/// Maximum number of allowed nodes stored, after this the prefetching is suspended
	u_int64_t maxSize;

	int storedNodes;
	unsigned int iterationNo;

        int lastFinishedChunk;

	/// number of acknowledgements awaited before deleting the chunk of the tree
        int *chunkAck;

	map<CacheKey,ParticleCacheEntry*> *particleCacheTable;
	int storedParticles;
	//bool proxyInitialized; // checks if the streaming proxy has been delegated or not
	map<MapKey,int> outStandingRequests;
	map<CacheKey,int> outStandingParticleRequests;
        map<NodeCacheEntry*,int> *delayedRequests;

	/******************************************************************
	 * Method section
	 ******************************************************************/

	/// Insert all nodes with root "node" coming from "from" into the nodeCacheTable.
	void addNodes(int chunk,int from,CacheNode *node);
	//void addNode(CacheKey key,int from,CacheNode &node);
	/// @brief Check all the TreePiece buckets which requested a node, and call
	/// them back so that they can continue the treewalk.
	void processRequests(int chunk,CacheNode *node,int from,int depth);
	/// @brief Fetches the Node from the correct TreePiece. If the TreePiece
	/// is in the same processor fetch it directly, otherwise send a message
	/// to the remote TreePiece::fillRequestNode
	CacheNode *sendNodeRequest(int chunk,NodeCacheEntry *e,int reqID);
	ExternalGravityParticle *sendParticleRequest(ParticleCacheEntry *e,int reqID);

#ifdef CACHE_TREE
	/// Construct a tree based on the roots given as input recursively. The
	/// tree will be a superset of all the trees given. Only the mininum
	/// number of nodes is duplicated.
	/// @return the root of the global tree
	GenericTreeNode *buildProcessorTree(int n, GenericTreeNode **gtn);

	/// Fill a hashtable (chunkRootTable) with an entry per chunkRoot, so
	/// later TreePieces can look them up
	/// @param node the current node in the recursion
	/// @param keys the list of keys that have to be mapped
	/// @return the number of keys mapped by the call
	int createLookupRoots(GenericTreeNode *node, Tree::NodeKey *keys);
#endif

 public:

	CacheManager(int size);
	CacheManager(CkMigrateMessage *);
	~CacheManager(){};

	/** @brief Called by TreePiece to request for a particular node. It can
	 * return the node if it is already present in the cache, or call
	 * sendNodeRequest to get it. Returns null if the Node has to come from
	 * remote.
	*/
	CacheNode *requestNode(int requestorIndex, int remoteIndex, int chunk, CacheKey key, int reqID, bool isPrefetch);
	// Shortcut for the other recvNodes, this receives only one node
	//void recvNodes(CacheKey ,int ,CacheNode &);
	/** @brief Receive the nodes incoming from the remote
	 * TreePiece::fillRequestNode. It imports the nodes into the cache and
	 * process all pending requests.
	 */
	void recvNodes(FillNodeMsg *msg);
	
	ExternalGravityParticle *requestParticles(int requestorIndex, int chunk, const CacheKey key, int remoteIndex, int begin, int end, int reqID, bool isPrefetch);
	//void recvParticles(CacheKey key,GravityParticle *part,int num, int from);
        void recvParticles(FillParticleMsg *msg);

#ifdef CACHE_TREE
	/** Convert a key into a node in the cache internal tree (the one built
	 *  on top of all the TreePieces in this processor
	 */
	inline GenericTreeNode *chunkRootToNode(const Tree::NodeKey k) {
	  NodeLookupType::iterator iter = chunkRootTable.find(k);
	  if (iter != chunkRootTable.end()) return iter->second;
	  else return NULL;
	}
	/** Return the root of the cache internal tree */
	inline GenericTreeNode *getRoot() { return root; }
#endif

	/** Invoked from the mainchare to start a new iteration. It calls the
	    prefetcher of all the chare elements residing on this processor, and
	    send a message to them to start the local computation. At this point
	    it is guaranteed that all the chares have already called
	    markPresence, since that is done in ResumeFromSync, and cacheSync is
	    called only after a reduction from ResumeFromSync.
	 */
	void cacheSync(double theta, int activeRung, const CkCallback& cb);

	/** Inform the CacheManager that the chare element will be resident on
	    this processor for the next iteration. It is done after treebuild in
	    the first iteration (when looking up localCache), and then it is
	    maintained until migration. Since this occur only with loadbalancer,
	    the unregistration and re-registration is done right before AtSync
	    and after ResumeFromSync.
	 */
	int markPresence(int index, GenericTreeNode *root);
	/// Before changing processor, chares deregister from the CacheManager
	void revokePresence(int index);

	/// Called from the TreePieces to acknowledge that a particular chunk
	/// has been completely used, and can be deleted
	void finishedChunk(int num, u_int64_t weight);
        /// Called from the TreePieces to acknowledge that they have completely
        /// finished their computation
        void allDone();
        void recordCommunication(int receiver, CommData *data);

	/// Collect the statistics for the latest iteration
	void collectStatistics(CkCallback& cb);

        void stopHPM(CkCallback& cb) {
#ifdef HPM_COUNTER
          hpmTerminate(1);
#endif
          contribute(0, 0, CkReduction::concat, cb);
        }

	map<int,GenericTreeNode*> *getRegisteredChares() {
	  return &registeredChares;
	}
};

extern CProxy_CacheManager cacheManagerProxy;
#endif

