/*
 * File: CBSolverPluginFactory.h
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


#ifndef CB_SOLVER_PLUGIN_FACTORY
#define CB_SOLVER_PLUGIN_FACTORY

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ParameterMap.h"

class CBSolverPlugin;


class CBSolverPluginFactory
{
public:
    using FactoryFunction = std::function<CBSolverPlugin *()>;
    
    CBSolverPluginFactory();
    
    /// Creates every plugin enabled in the parameters, in producer order. Ownership passes to the caller.
    std::vector<std::unique_ptr<CBSolverPlugin>> LoadAllPlugins(ParameterMap* parameters);
    
protected:
private:
    /// Parameter key and producer for each plugin. A sequence rather than a map because the
    /// order is load-bearing: CBSolver::PrepareSimulation initialises the plugins in it.
    std::vector<std::pair<std::string, FactoryFunction>> producers_;
};

#endif
