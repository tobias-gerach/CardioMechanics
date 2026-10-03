/*
 * File: CBRobinBoundary.h
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


#ifndef CB_ROBIN_BOUNDARY
#define CB_ROBIN_BOUNDARY

#include <map>
#include <memory>

#include "CBSolverPlugin.h"
#include "CBElementSurfaceT3.h"

using namespace math_pack;

/// A spring and dashpot on the centroid of each surface triangle. RobinBoundary lets them act along
/// the reference normal only, RobinBoundaryGeneral on the full displacement and velocity; the two
/// keep their own XML keys and default stiffness so that existing input files load unchanged.
class CBRobinBoundary : public CBSolverPlugin {
public:
    explicit CBRobinBoundary(bool projectOnNormal);
    ~CBRobinBoundary() {}
    
    std::string GetName() override { return key_; }
    
    /// fundamental CBSolverPlugin functions
    void Init() override;
    void Apply(PetscScalar time) override;
    void ApplyToNodalForces() override;
    void ApplyToNodalForcesJacobian() override;
    void StepBack() override;
    void ReferenceChanged() override;
    void Export(TFloat time) override;
    void WriteToFile(TFloat time) override;
    void Prepare() override;
    
    CBStatus GetStatus() override {return status_;}
    
protected:
private:
    const bool projectOnNormal_;
    const std::string key_;
    TFloat dt_, prevDt_;
    typedef CBSolverPlugin Base;
    static constexpr int numNodes = CBElementSurfaceT3::numNodes;
    std::vector<CBElementSurfaceT3 *> contactSurfaceElements_;
    std::vector<Vector3<TFloat>> referenceNormals_;
    std::vector<Vector3<TFloat>> initialPos_;
    std::vector<Vector3<TFloat>> displacement_;
    std::vector<Vector3<TFloat>> prevDisplacement_;
    std::vector<Vector3<TFloat>> velocity_;
    
    CBStatus status_ = CBStatus::DACCORD;
    
    bool isFirstStep_ = true;
    bool stepBack_ = false;
    bool hasStarted_ = false;
    
    /// functions
    void InitContactSurfaces();
    Vector3<TFloat> Project(const Vector3<TFloat> &x, const Vector3<TFloat> &N) const;
    void CalcForceContributionOfElement(Vector3<TFloat> u, Vector3<TFloat> v, CBElementSurfaceT3 *triangle,
                                        const Vector3<TFloat> &refNormalVector,
                                        TFloat *nodalForces);
    
    /// xml parameters
    /// general options
    TFloat startTime_;
    bool export_;
    
    /// boundary model specific options
    TInt normalVectorSign_ = 1;
    PetscScalar alpha_;
    PetscScalar beta_;
    PetscInt surfaceIndex_;
    std::string filename_;
    std::ofstream file_;
    
    
    // ----- Values needed for export -----
    
    std::vector<Vector3<TFloat>> ContactForces_;
};
#endif // ifndef CB_ROBIN_BOUNDARY
