/*
 * File: CBElementSurface.h
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


#ifndef CB_ELEMENT_SURFACE_H
#define CB_ELEMENT_SURFACE_H

#include "CBElement.h"

using namespace math_pack;

/// What the settings declare a surface for. It is independent of the element's shape, whose node
/// count comes from the mesh, so the model and the plugins select their surfaces by role.
enum class SurfaceRole {
    None,
    Cavity,
    Robin,
    ContactMaster,
    ContactSlave
};

class CBElementSurface : public CBElement {
public:
    CBElementSurface() {
        SetProperty(ElementProperty::Surface);
    }
    
    CBElementSurface(CBElementSurface &other) : CBElement(other) {
        surfaceIndex_ = other.surfaceIndex_;
        surfaceElementIndex_ = other.surfaceElementIndex_;
        initialArea_ = other.initialArea_;
        surfaceTractionScaling_ = other.surfaceTractionScaling_;
        role_ = other.role_;
    }
    
    SurfaceRole GetRole() const {return role_;}
    
    void SetRole(SurfaceRole role) {role_ = role;}
    
    TFloat GetSurfaceTractionScaling() {return surfaceTractionScaling_;}
    
    void SetSurfaceTractionScaling(TFloat surfaceTractionScaling) {surfaceTractionScaling_ = surfaceTractionScaling;}
    
    TInt GetSurfaceIndex() {return surfaceIndex_;}
    
    void SetSurfaceIndex(TInt surfaceIndex) {surfaceIndex_ = surfaceIndex;}
    
    void SetSurfaceElementIndex(TInt index) {surfaceElementIndex_ = index;}
    
    TInt GetSurfaceElementIndex() {return surfaceElementIndex_;}
    
    virtual ~CBElementSurface() {}
    
    TFloat GetInitialArea() {return initialArea_;}
    
    virtual void SetWeight(unsigned int i, TFloat w) {}
    
    virtual TFloat GetWeight(unsigned int i) {return 1.0;}
    
    virtual void GetAreaNormalAndIndices(TFloat *c, TInt *nodesCoordsIndices) {
        throw std::runtime_error("CBElementSurface::GetAreaNormalAndIndices s not implemented!");
    }
    
protected:
    TInt surfaceIndex_; // Index of the respective surface ( contains several elements)
    TInt surfaceElementIndex_; // Index of the element in the surface file -> not the element index !!!!
    TInt initialArea_;
    TFloat surfaceTractionScaling_ = 1.0; // individual scaling for boundary conditions
    SurfaceRole role_ = SurfaceRole::None;
    
private:
    CBElementSurface(const CBElementSurface &);
    void operator=(const CBElementSurface &);
};
#endif // ifndef CB_ELEMENT_SURFACE_H
