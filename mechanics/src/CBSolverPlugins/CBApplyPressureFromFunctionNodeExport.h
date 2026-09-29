/*
 * File: CBApplyPressureFromFunctionNodeExport.h
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


#ifndef CB_APPLY_PRESSURE_FROM_FUNCTION_NODE_EXPORT
#define CB_APPLY_PRESSURE_FROM_FUNCTION_NODE_EXPORT

#include "CBApplyPressureFromFunction.h"

/// ApplyPressureFromFunction that stops the run once a cavity volume or a time is reached, and
/// writes the deformed nodes of that step as a tetgen node file.
class CBApplyPressureFromFunctionNodeExport : public CBApplyPressureFromFunction
{
public:
    ~CBApplyPressureFromFunctionNodeExport() override;
    
    void Init() override;
    bool ExitCheck(TFloat time) override;
    std::string GetName() override { return("ApplyPressureFromFunctionNodeExport"); }
    
private:
    void ExportNodeFile();
    
    TInt stopSurface_;
    TFloat stopVolume_;
    TFloat nodeExportTime_;
    std::string nodeFilename_;
    Vec coords_ = nullptr;
    Vec coordsSeq_ = nullptr;
    VecScatter coordsScatter_ = nullptr;
};

#endif
