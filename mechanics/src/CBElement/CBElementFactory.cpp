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
/// A type that names a shape fixes the node count itself.
CBElementFactory::FactoryFunction Shape(CBElement *(*make)()) {
    return [make](unsigned int) {return make();};
}

/// A role whose plugins integrate six-node faces is carried by a triangle of the face's node count.
/// The others are carried by three-node triangles, and the loader refines a six-node face declared
/// with one of them into four.
CBElementFactory::FactoryFunction SurfaceWithRole(SurfaceRole role, bool sixNodeFaces) {
    return [role, sixNodeFaces](unsigned int numNodes) {
        CBElementSurface *surface = nullptr;
        if (sixNodeFaces && numNodes == 6)
            surface = new CBElementSurfaceT6;
        else
            surface = new CBElementSurfaceT3;
        surface->SetRole(role);
        return surface;
    };
}
}


CBElementFactory::CBElementFactory() {
    producers_["T4"]      = Shape(CBElementSolidT4::New);
    producers_["T4MINI"]  = Shape(CBElementSolidT4Mini::New);
    producers_["T10"]     = Shape(CBElementSolidT10::New);
    producers_["T10P1"]   = Shape(CBElementSolidT10P1::New);
    producers_["T3"]      = Shape(CBElementSurfaceT3::New);
    producers_["T6"]      = Shape(CBElementSurfaceT6::New);
    producers_["CAVITY"]          = SurfaceWithRole(SurfaceRole::Cavity, true);
    producers_["CONTACT_ROBIN"]   = SurfaceWithRole(SurfaceRole::Robin, false);
    producers_["CONTACT_MASTER"]  = SurfaceWithRole(SurfaceRole::ContactMaster, false);
    producers_["CONTACT_SLAVE"]   = SurfaceWithRole(SurfaceRole::ContactSlave, false);
}

CBElement *CBElementFactory::New(std::string elementType, unsigned int numNodes) {
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
                                 + "\t CAVITY\t:3- or 6-node triangle surface element for circulatory system plugin\n"
                                 + "\t T3\t: 3-node triangle surface element\n"
                                 + "\t T6\t: 6-node triangle surface element\n");
    } else {
        result = it->second(numNodes);
    }
    return result;
}
