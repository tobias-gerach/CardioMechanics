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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ParameterMap.h"

class CBSolverPlugin;


class CBSolverPluginFactory
{
public:
    using FactoryFunction = std::function<CBSolverPlugin *()>;
    
    CBSolverPluginFactory();
    
    /// Creates a plugin of the given type. Ownership passes to the caller.
    std::unique_ptr<CBSolverPlugin> New(const std::string& pluginName);
    
    /// Creates every plugin enabled in the parameters, in the fixed order the solver
    /// initializes them in. Ownership passes to the caller.
    std::vector<std::unique_ptr<CBSolverPlugin>> LoadAllPlugins(ParameterMap* parameters);
    
protected:
private:
    std::map<std::string, FactoryFunction> producers_;
};

#endif
