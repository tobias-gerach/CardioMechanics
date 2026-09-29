/*
 * File: CBContactHandling.h
 *
 * Institute of Biomedical Engineering, 
 * Karlsruhe Institute of Technology (KIT)
 * https://www.ibt.kit.edu
 * 
 * Repository: https://github.com/KIT-IBT/CardioMechanics
 *
 * License: GPL-3.0 (See accompanying file LICENSE or visit https://www.gnu.org/licenses/gpl-3.0.html)
 *
 */


#ifndef CB_CONTACT_HANDLING_H
#define CB_CONTACT_HANDLING_H

#include <array>
#include <map>
#include <set>
#include "Matrix3.h"

#include "CBSolverPlugin.h"

#include "CBElementSurfaceT3.h"

using namespace math_pack;

class CBContactHandling : public CBSolverPlugin {
public:
    CBContactHandling();
    virtual ~CBContactHandling() {}
    
    void Init() override;
    void Apply(PetscScalar time) override;
    void ApplyToNodalForces() override;
    void ApplyToNodalForcesJacobian() override;
    void StepBack() override;
    bool WantsToAnalyzeResults() override {return true; }
    void AnalyzeResults() override;
    void Export(TFloat time) override;
    void WriteToFile(TFloat time) override;
    
    void SetAlpha(TFloat a) {alpha_ = a; }
    
    TFloat GetAlpha() {return alpha_; }
    
    void           GetMasterNodesDistancesToSlaveElements(Vec *d);
    std::set<TInt> GetMasterNodesLocalIndices();
    std::set<TInt> GetMasterWithSlaveNodesLocalIndices();
    void           Prepare() override;
    double         GetPreparationProgress() override;
    
    std::string GetName() override {return "ContactHandling"; }
    
protected:
private:
    friend class CBParameterEstimator;
    
    /// Contact is formulated on three-node triangles, with one Gauss point near each vertex.
    static constexpr int numNodes       = CBElementSurfaceT3::numNodes;
    static constexpr int numGaussPoints = 3;
    
    /// A master element with the slave element found at each of its Gauss points and vertices, -1
    /// where there is none, and the gap vector to it.
    struct MasterElement {
        explicit MasterElement(CBElementSurfaceT3 *e) : element(e) {
            slaveAtGaussPoint.fill(-1);
            slaveAtVertex.fill(-1);
        }
        
        CBElementSurfaceT3 *element;
        std::array<TInt, numGaussPoints> slaveAtGaussPoint;
        std::array<TInt, numNodes> slaveAtVertex;
        std::array<Vector3<TFloat>, numGaussPoints> distanceVectorToSlave;
        TFloat distanceToSlave = 0;
    };
    
    void DetermineSlaveNodes();
    void DetermineInitialSlaveElementsAtGaussPoints();
    bool CheckIfSlave(TFloat *slaveNodes, int slaveInd, Vector3<TFloat> *p, Vector3<TFloat> *nv, TFloat &dist);
    int  SearchForSlave(TFloat *slaveNodes, int oldSlave, Vector3<TFloat> *p, Vector3<TFloat> *nv, TFloat &dist);
    void DetermineSlaveElementsAtGaussPoints();
    void DetermineSlaveElementsAtVertices();
    void LoadMasterElements();
    void LoadSlaveElements();
    void UpdateDistancesMasterSlave();
    /// Contact pressure of master element i, negative where the contact force points along the slave normal.
    PetscScalar ContactPressure(size_t i) const;
    void CalcContributionToContactForceAtGaussPoint(const Triangle<TFloat> &masterTriangle,
                                                    const Triangle<TFloat> &slaveTriangle, TInt gaussPointIndex,
                                                    TFloat *nodalForces, TFloat *distances, TFloat scaling);
    Vector3<TFloat> CalculateDistanceAtGaussPoint(const Triangle<TFloat> &masterTriangle,
                                                  const Triangle<TFloat> &slaveTriangle, TInt gaussPointIndex);
    Vector3<TFloat> CalculateDistanceAtVertex(const Triangle<TFloat> &masterTriangle,
                                              const Triangle<TFloat> &slaveTriangle, TInt index);
    std::vector<MasterElement>        masterElements_;
    std::vector<CBElementSurfaceT3 *> slaveElements_;
    
    // -----
    std::map<int, std::vector<int> *> slaveNeighbors_;
    int maxDepth_ = 10;
    
    // -----
    
    Vec slaveElementsNodes_;
    Vec masterNodesDistToSlaveElements_;
    std::set<TInt> masterNodesLocalIndices_;
    std::vector<TInt> slaveElementsSurfaceIndices_;
    Vec slaveElementsNodesSeq_;
    
    std::vector<TInt> slaveElementsNodesIndicesGlobal_;
    
    PetscInt numGlobalSlaveElements_;
    
    VecScatter scatter_;
    
    std::string filename_;
    std::string initType_;
    std::string surfaceNormalDirection_;
    TFloat dt_, prevDt_;
    PetscScalar maxDistanceToSlave_;
    PetscScalar transitionDistance_;
    PetscScalar maxAngle_;
    PetscScalar alpha_;
    PetscScalar beta_;
    PetscScalar maxAlpha_;
    PetscScalar cntMax_;
    PetscInt    cnt_;
    PetscScalar lastTime_  = 0;
    PetscScalar time_      = 0;
    TInt normalVectorSign_ = 1;
    TInt oneWayForce_;
    
    TFloat startTime_;
    bool   hasStarted_ = false;
    
    bool useContactHandlingToFitPeri_ = false;
    
    bool isFirstStep_ = true;
    bool stepBack_    = false;
    
    // ----- Values needed for export -----
    
    std::vector<Vector3<TFloat>> masterContactForces_;
    std::vector<Vector3<TFloat>> masterCorrespondingSlaveNormal_;
    std::vector<Vector3<TFloat>> masterContactDistances_;
    std::vector<TInt> masterCorrespondingSlaveFound_;
    PetscScalar averageDist_;
    PetscScalar averageContactPressure_;
    PetscScalar globalAverageDist_ = 0;  // only process zero !!!
    PetscScalar globalAverageContactPressure_ = 0;  // only process zero !!!
    bool export_;
    
    typedef CBSolverPlugin Base;
};
#endif  // ifndef CB_CONTACT_HANDLING_H
