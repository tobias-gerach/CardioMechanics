/*
 * File: CBApplyPressureFromFunctionNodeExport.cpp
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

#include "CBApplyPressureFromFunctionNodeExport.h"
#include "CBSolver.h"

CBApplyPressureFromFunctionNodeExport::~CBApplyPressureFromFunctionNodeExport() {
    VecScatterDestroy(&coordsScatter_);
    VecDestroy(&coordsSeq_);
    VecDestroy(&coords_);
}

void CBApplyPressureFromFunctionNodeExport::Init() {
    CBApplyPressureFromFunction::Init();
    
    const std::string key = "Plugins." + GetName();
    stopSurface_ = parameters_->Get<TInt>(key + ".StopSurface", -1);
    if (stopSurface_ != -1) {
        // Volumes are computed only for the surfaces the groups load.
        if (valuesStructs_.find(stopSurface_) == valuesStructs_.end())
            throw std::runtime_error("CBApplyPressureFromFunctionNodeExport::Init(): StopSurface " +
                                     std::to_string(stopSurface_) + " is not loaded by any group.");
        stopVolume_ = parameters_->Get<TFloat>(key + ".StopVolume");
    } else
        nodeExportTime_ = parameters_->Get<TFloat>(key + ".NodeExportTime");
    nodeFilename_ = parameters_->Get<std::string>(key + ".NodeExportFile");
    
    DCCtrlPETSc::CreateVector(3*adapter_->GetSolver()->GetNumberOfLocalNodes(), PETSC_DETERMINE, &coords_);
    if (DCCtrl::IsParallel())
        VecScatterCreateToZero(coords_, &coordsScatter_, &coordsSeq_);
}

/// Called once per committed step, so the node file holds the geometry of the last step computed.
/// The export time is taken to be reached within half a step before it, as the ends of the
/// pressure intervals are, so that it fires even if no step lands on it.
bool CBApplyPressureFromFunctionNodeExport::ExitCheck(TFloat time) {
    const bool reached = stopSurface_ == -1
        ? time >= nodeExportTime_ - 0.5*adapter_->GetSolver()->GetTiming().GetTimeStep()
        : valuesStructs_.at(stopSurface_).volume*1e6 <= stopVolume_;
    if (reached)
        ExportNodeFile();
    return reached;
}

void CBApplyPressureFromFunctionNodeExport::ExportNodeFile() {
    adapter_->GetSolver()->GetNodeCoordinates(coords_);
    
    if (DCCtrl::IsParallel()) {
        // Gather to process zero
        VecScatterBegin(coordsScatter_, coords_, coordsSeq_, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(coordsScatter_, coords_, coordsSeq_, INSERT_VALUES, SCATTER_FORWARD);
    }
    
    if (DCCtrl::IsProcessZero()) {
        TInt numNodes = adapter_->GetSolver()->GetNumberOfNodes();
        TInt numCoords = 3*numNodes;
        
        ////// Get node coords //////
        
        std::vector<PetscInt> pos(numCoords);
        for (PetscInt i = 0; i < numCoords; i++)
            pos[i] = i;
        
        std::vector<PetscScalar> values(numCoords);
        if (DCCtrl::IsParallel())
            VecGetValues(coordsSeq_, numCoords, pos.data(), values.data());
        else
            VecGetValues(coords_, numCoords, pos.data(), values.data());
        
        ////// Get boundary conditions //////
        
        auto bc = std::make_unique<bool[]>(numCoords);
        adapter_->GetNodesComponentsBoundaryConditionsGlobal(numCoords, pos.data(), bc.get());
        
        ////// Write node file //////
        
        std::ofstream nodeFile(nodeFilename_);
        if (!nodeFile.good())
            throw std::runtime_error(
                                     "CBApplyPressureFromFunctionNodeExport::ExportNodeFile: Couldn't create " + nodeFilename_ + ".");
        
        nodeFile << numNodes << " 3 1 0" << std::endl;
        
        for (PetscInt i = 0; i < numNodes; i++) {
            TFloat x = 1e3*values[3*i];
            TFloat y = 1e3*values[3*i+1];
            TFloat z = 1e3*values[3*i+2];
            TInt   b = bc[3*i] + (bc[3*i+1] << 1) + (bc[3*i+2] << 2); // 001 (1): x fixed; 010 (2): y fixed; 100 (4): z fixed; 111 (7): x,y,z fixed
            nodeFile << i+1 << " " << x << " " << y << " " << z << " " << b << std::endl;
        }
        
        nodeFile.close();
    }
    DCCtrl::print << "Exported nodes to " << nodeFilename_ << std::endl;
} // CBApplyPressureFromFunctionNodeExport::ExportNodeFile
