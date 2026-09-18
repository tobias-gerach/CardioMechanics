/*
 * File: CBSolverPluginFactory.cpp
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


#include "CBSolverPlugin.h"
#include "CBSolverPluginFactory.h"

#include "CBContactHandling.h"
#include "CBApplyPressureFromFunction.h"
#include "CBApplyPressureFromFunctionNodeExport.h"
#include "CBCirculation.h"
#include "CBApplyPressure.h"
#include "CBReferenceRecovery.h"
#include "CBLoadUnloadedState.h"
#include "CBRobinBoundary.h"
#include "CBRobinBoundaryGeneral.h"
#include "CBacCELLerate.h"
#include "CBPointsCtrl.h"

/// The keys are the parameter keys, which for some plugins differ from GetName().
CBSolverPluginFactory::CBSolverPluginFactory() : producers_{
    {"acCELLerate",                         []() { return new CBacCELLerate(); }},
    {"LoadUnloadedState",                   []() { return new CBLoadUnloadedState(); }},
    {"ReferenceRecovery",                   []() { return new CBReferenceRecovery(); }},
    {"Circulation",                         []() { return new CBCirculation(); }},
    {"ContactHandling",                     []() { return new CBContactHandling(); }},
    {"RobinBoundary",                       []() { return new CBRobinBoundary(); }},
    {"RobinBoundaryGeneral",                []() { return new CBRobinBoundaryGeneral(); }},
    {"ApplyPressureFromFunction",           []() { return new CBApplyPressureFromFunction(); }},
    {"ApplyPressureFromFunctionNodeExport", []() { return new CBApplyPressureFromFunctionNodeExport(); }},
    {"ApplyPressure",                       []() { return new CBApplyPressure(); }},
    {"PointsCtrl",                          []() { return new CBPointsCtrl(); }},
} {}

std::vector<std::unique_ptr<CBSolverPlugin>> CBSolverPluginFactory::LoadAllPlugins(ParameterMap *parameters) {
    std::vector<std::unique_ptr<CBSolverPlugin>> plugins;
    
    for (auto &[key, produce] : producers_) {
        if (!parameters->Get<bool>("Solver.Plugins." + key, false))
            continue;
        plugins.emplace_back(produce());
        plugins.back()->SetParameters(parameters);
    }
    
    return plugins;
}
