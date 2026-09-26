/*
 * File: CBElementFactory.cpp
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


#include "CBElementFactory.h"

#include "CBElement.h"
#include "CBElementSolid.h"
#include "CBElementSolidT4.h"
#include "CBElementSolidT4Mini.h"
#include "CBElementSolidT10.h"
#include "CBElementSolidT10P1.h"
#include "CBElementSurface.h"
#include "CBElementSurfaceT3.h"
#include "CBElementSurfaceT6.h"

namespace {
/// Roles are carried by three-node triangles until their plugins support six-node faces; the
/// loader refines a six-node face declared with a role into four of them.
CBElementFactory::FactoryFunction TriangleWithRole(SurfaceRole role) {
    return [role] {
        auto *surface = new CBElementSurfaceT3;
        surface->SetRole(role);
        return surface;
    };
}
}


CBElementFactory::CBElementFactory() {
    producers_["T4"]      = CBElementSolidT4::New;
    producers_["T4MINI"]  = CBElementSolidT4Mini::New;
    producers_["T10"]     = CBElementSolidT10::New;
    producers_["T10P1"]   = CBElementSolidT10P1::New;
    producers_["T3"]      = CBElementSurfaceT3::New;
    producers_["T6"]      = CBElementSurfaceT6::New;
    producers_["CAVITY"]          = TriangleWithRole(SurfaceRole::Cavity);
    producers_["CONTACT_ROBIN"]   = TriangleWithRole(SurfaceRole::Robin);
    producers_["CONTACT_MASTER"]  = TriangleWithRole(SurfaceRole::ContactMaster);
    producers_["CONTACT_SLAVE"]   = TriangleWithRole(SurfaceRole::ContactSlave);
}

CBElement *CBElementFactory::New(std::string elementType) {
    CBElement *result = nullptr;
    
    auto it = producers_.find(elementType);
    
    if (it == producers_.end()) {
        throw std::runtime_error(std::string("Unkown element type: [" + elementType + "] You might have to extend CBElementFactory \n Available elements are: \n")
                                 + "\t T4\t:4-node Iso-P1 tetrahedral element,\n"
                                 + "\t T4MINI\t:4-node Iso-P1 tetrahedral element enriched by a condensed bubble, with a linear pressure field on its 4 vertices (MINI),\n"
                                 + "\t T10\t:10-node Iso-P2 tetrahedral element,\n"
                                 + "\t T10P1\t:10-node Iso-P2 tetrahedral element with a linear pressure field on its 4 vertices (P2P1 Taylor-Hood),\n"
                                 + "\t CONTACT_MASTER\t:3-node triangle surface element for contact problems\n"
                                 + "\t CONTACT_SLAVE\t:3-node triangle surface element for contact problems\n"
                                 + "\t CONTACT_ROBIN\t:3-node triangle surface element for Robin boundary condition\n"
                                 + "\t CAVITY\t:3-node triangle surface element for circulatory system plugin\n"
                                 + "\t T3\t: 3-node triangle surface element\n"
                                 + "\t T6\t: 6-node triangle surface element\n");
    } else {
        result = it->second();
    }
    return result;
}
