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

#include <stdexcept>

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

namespace {
/// The order in which the solver creates and initializes the plugins. It is load-bearing:
/// CBSolver::PrepareSimulation prepares them one after the other in this sequence.
/// The keys are the parameter keys, which for some plugins differ from GetName().
const std::vector<std::string> kPluginOrder = {
    "acCELLerate",
    "LoadUnloadedState",
    "ReferenceRecovery",
    "Circulation",
    "ContactHandling",
    "RobinBoundary",
    "RobinBoundaryGeneral",
    "ApplyPressureFromFunction",
    "ApplyPressureFromFunctionNodeExport",
    "ApplyPressure",
    "PointsCtrl",
};
}

CBSolverPluginFactory::CBSolverPluginFactory() {
    producers_["acCELLerate"]           = []() { return new CBacCELLerate(); };
    producers_["LoadUnloadedState"]     = []() { return new CBLoadUnloadedState(); };
    producers_["ReferenceRecovery"]     = []() { return new CBReferenceRecovery(); };
    producers_["Circulation"]           = []() { return new CBCirculation(); };
    producers_["ContactHandling"]       = []() { return new CBContactHandling(); };
    producers_["RobinBoundary"]         = []() { return new CBRobinBoundary(); };
    producers_["RobinBoundaryGeneral"]  = []() { return new CBRobinBoundaryGeneral(); };
    producers_["ApplyPressureFromFunction"] = []() { return new CBApplyPressureFromFunction(); };
    producers_["ApplyPressureFromFunctionNodeExport"] = []() { return new CBApplyPressureFromFunctionNodeExport(); };
    producers_["ApplyPressure"]         = []() { return new CBApplyPressure(); };
    producers_["PointsCtrl"]            = []() { return new CBPointsCtrl(); };
}

std::unique_ptr<CBSolverPlugin> CBSolverPluginFactory::New(const std::string& pluginName) {
    auto it = producers_.find(pluginName);
    
    if (it == producers_.end()) {
        std::string available;
        for (auto &producer : producers_)
            available += "\t" + producer.first + "\n";
        throw std::runtime_error("Unknown solver plugin: [" + pluginName +
                                 "] You might have to extend CBSolverPluginFactory \n Available plugins are: \n" + available);
    }
    return std::unique_ptr<CBSolverPlugin>(it->second());
}

std::vector<std::unique_ptr<CBSolverPlugin>> CBSolverPluginFactory::LoadAllPlugins(ParameterMap *parameters) {
    std::vector<std::unique_ptr<CBSolverPlugin>> plugins;
    
    for (auto &key : kPluginOrder) {
        if (!parameters->Get<bool>("Solver.Plugins." + key, false))
            continue;
        auto plugin = New(key);
        plugin->SetParameters(parameters);
        plugins.push_back(std::move(plugin));
    }
    
    return plugins;
}
