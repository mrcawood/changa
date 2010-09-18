/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _ORB3DLB_H_
#define _ORB3DLB_H_

#include "CentralLB.h"
#include "MapStructures.h"
#include "Orb3dLB.decl.h"
#include "TaggedVector3D.h"
#include <queue>

void CreateOrb3dLB();
BaseLB * AllocateOrb3dLB();

class Orb3dLB : public CentralLB {
  friend class MultistepLB;
private:
  CmiBool firstRound; 
  CmiBool centroidsAllocated;
  ComparatorFn compares[NDIMS];
  ComparatorFn pc[NDIMS];
  // pointer to stats->to_proc
  CkVec<int> *mapping;

  int procsPerNode;

  // things are stored in here before work
  // is ever called.
  TaggedVector3D *tpCentroids;
  CkReductionMsg *tpmsg;
  int nrecvd;
  bool haveTPCentroids;

  CmiBool QueryBalanceNow(int step);

  void printData(BaseLB::LDStats &stats, int phase, int *revObjMap);

public:
  Orb3dLB(const CkLBOptions &);
  Orb3dLB(CkMigrateMessage *m):CentralLB(m) { lbname = "Orb3dLB"; }
  void work(BaseLB::LDStats* stats);
  void receiveCentroids(CkReductionMsg *msg);
  void directMap(TPObject *tp, int ntp, Node *nodes);
  void map(TPObject *tp, int ntp, int nn, Node *procs, int xs, int ys, int zs, int dim);
  int nextDim(int dim, int xs, int ys, int zs);
  TPObject *partitionEvenLoad(TPObject *tp, int &ntp);
  Node *halveNodes(Node *start, int np);


};

#endif /* _ORB3DLB_H_ */

/*@}*/