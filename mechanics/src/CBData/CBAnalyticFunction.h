/*
 * File: CBAnalyticFunction.h
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


#pragma once

#include "DCType.h"
#include "ParameterMap.h"
#include "CBTensionModel.h"

#include <stdint.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>

/// a general definition of time-dependent analytic function
/// note: when adding a new class here, it has to be added to
/// CBDataFromFunction.cpp (and CBTensionFromFunction) as well
class CBAnalyticFunction {
public:
    CBAnalyticFunction() {}
    
    virtual ~CBAnalyticFunction() {}
    
    virtual TFloat operator()(TFloat time) = 0;
    CBAnalyticFunction(ParameterMap *parameters, std::string parameterKey) {
        startTime_ = parameters->Get<TFloat>(parameterKey + ".StartTime", 0);
        stopTime_  = parameters->Get<TFloat>(parameterKey + ".StopTime", 0);
        keepMaxValue_ = parameters->Get<bool>(parameterKey + ".KeepMaxValue", true);
    }
    
protected:
    TFloat startTime_;
    TFloat stopTime_;
    bool keepMaxValue_;
};


/// a function increasing linearly between startTime and stopTime from 0 to 1
class Linear : public CBAnalyticFunction {
public:
    virtual ~Linear() {}
    
    TFloat operator()(TFloat time) override {
        TFloat val;
        
        if (time < CBAnalyticFunction::startTime_) {
            val = 0.0;
        } else if (time < CBAnalyticFunction::stopTime_+0.5*(time-prevTime_)) {
            val = (time - CBAnalyticFunction::startTime_) * 1.0 /
            (CBAnalyticFunction::stopTime_ - CBAnalyticFunction::startTime_);
        } else if (CBAnalyticFunction::keepMaxValue_) {
            val = 1.0;
        } else {
            val = 0.0;
        }
        prevTime_ = time;
        return val;
    }
    
    Linear(ParameterMap *parameters, std::string parameterKey) : CBAnalyticFunction(parameters, parameterKey) {}
    
protected:
    TFloat prevTime_ = 0.0;
};


/// a function defining a sinus arc, uses frequency omega and phase phi
class Sinus : public CBAnalyticFunction {
public:
    virtual ~Sinus() {}
    
    TFloat operator()(TFloat time) override {
        if ((time >= CBAnalyticFunction::startTime_) && (time <= CBAnalyticFunction::stopTime_))
            return 0.5 - 0.5 * cos(omega_ * (time - CBAnalyticFunction::startTime_) + phi_);
        else
            return 0;
    }
    
    Sinus(ParameterMap *parameters, std::string parameterKey) : CBAnalyticFunction(parameters, parameterKey) {
        omega_ = parameters->Get<TFloat>(parameterKey + ".Sinus.Omega", 0);
        phi_   = parameters->Get<TFloat>(parameterKey + ".Sinus.Phi", 0);
    }
    
protected:
    TFloat omega_ = 0;
    TFloat phi_ = 0;
};


/// a sin function for easier definition of only the positive sinus arc from startTime to StopTime
class SinusDriver : public CBAnalyticFunction {
public:
    virtual ~SinusDriver() {}
    
    TFloat operator()(TFloat time) override {
        if ((time >= CBAnalyticFunction::startTime_) && (time <= CBAnalyticFunction::stopTime_)) {
            return sin(3.14159* (time - CBAnalyticFunction::startTime_) /
                       (CBAnalyticFunction::stopTime_ - CBAnalyticFunction::startTime_) );
        } else {
            return 0;
        }
    }
    
    SinusDriver(ParameterMap *parameters, std::string parameterKey) : CBAnalyticFunction(parameters, parameterKey) {}
};


/// a function based on the time-dependent compliance of the left ventricle. Can be used e.g. for ventricular tension.
class DoubleHill : public CBAnalyticFunction {
    // See Stergiopulos 1996: http://pubmed.gov/8764256
    
public:
    virtual ~DoubleHill() {}
    
    TFloat operator()(TFloat time) override {
        if ((time >= CBAnalyticFunction::startTime_) && (time <= CBAnalyticFunction::stopTime_)) {
            TFloat t = time - onsetTime_;
            if (t > 0.0)
                t = fmod(t, period_);
            else
                t = 0.0;
            TFloat contrDriver = pow(t / contrTimeOffset_, contrRateConst_);
            contrDriver = contrDriver / (1.0 + contrDriver);                // Hill function
            TFloat relaxDriver = pow(t / relaxTimeOffset_, relaxRateConst_);
            relaxDriver = 1.0 / (1.0 + relaxDriver);                        // inverted Hill function
            TFloat driver = contrDriver * relaxDriver / driverMax_;
            return driver;
        } else {
            return 0.0;
        }
    }
    
    DoubleHill(ParameterMap *parameters, std::string parameterKey) : CBAnalyticFunction(parameters, parameterKey) {
        period_          = parameters->Get<TFloat>(parameterKey + ".DoubleHill.Period", 0.8);
        contrTimeOffset_ = parameters->Get<TFloat>(parameterKey + ".DoubleHill.ContrTimeOffset", 0.215);
        relaxTimeOffset_ = parameters->Get<TFloat>(parameterKey + ".DoubleHill.RelaxTimeOffset", 0.362);
        contrRateConst_  = parameters->Get<TFloat>(parameterKey + ".DoubleHill.ContrRateConst", 1.32);
        relaxRateConst_  = parameters->Get<TFloat>(parameterKey + ".DoubleHill.RelaxRateConst", 21.9);
        onsetTime_       = parameters->Get<TFloat>(parameterKey + ".DoubleHill.OnsetTime", 0.0);
        
        // calculate max value of contrDriver * relaxDriver needed for normalization
        std::vector<TFloat> driverVec;
        for (TInt i = 0; i <= 1000; i++) {
            TFloat t = i * period_ / 1000;
            TFloat contrDriver = pow(t / contrTimeOffset_, contrRateConst_);
            contrDriver = contrDriver / (1.0 + contrDriver);
            TFloat relaxDriver = pow(t / relaxTimeOffset_, relaxRateConst_);
            relaxDriver = 1.0 / (1.0 + relaxDriver);
            driverVec.push_back(contrDriver * relaxDriver);
        }
        driverMax_ = *std::max_element(begin(driverVec), end(driverVec));
    }
    
protected:
    TFloat period_;
    TFloat contrTimeOffset_;
    TFloat relaxTimeOffset_;
    TFloat contrRateConst_;
    TFloat relaxRateConst_;
    TFloat onsetTime_;
    TFloat driverMax_;
};


/// the cavity pressure of ODE (7) in Aróstica et al. 2025, doi:10.1016/j.cma.2024.117485, with p(0) = 0;
/// it is in Pa, so an ApplyPressureFromFunction interval using it sets Amplitude to 1
class BestelPressure : public CBAnalyticFunction {
public:
    virtual ~BestelPressure() {}
    
    /// Integrates from 0 on every call instead of stepping from the previous one, so that the
    /// shifted and repeated times of the generalized-alpha solver get the same value in any order.
    TFloat operator()(TFloat time) override {
        if (time <= 0)
            return 0.0;
        const TInt steps = static_cast<TInt>(std::ceil(time / maxSubStep_));
        const TFloat dt = time / steps;
        TFloat p = 0.0;
        for (TInt i = 0; i < steps; i++)
            p = CBTensionModel::RungeKutta4Step(p, i * dt, dt, [this](double t, double p) { return dpdt(t, p); });
        return p;
    }
    
    BestelPressure(ParameterMap *parameters, std::string parameterKey) : CBAnalyticFunction(parameters, parameterKey) {
        const std::string key = parameterKey + ".BestelPressure.";
        alphaMin_ = parameters->Get<TFloat>(key + "AlphaMin");
        alphaMax_ = parameters->Get<TFloat>(key + "AlphaMax");
        alphaPre_ = parameters->Get<TFloat>(key + "AlphaPre");
        alphaMid_ = parameters->Get<TFloat>(key + "AlphaMid");
        sigmaPre_ = parameters->Get<TFloat>(key + "SigmaPre");
        sigmaMid_ = parameters->Get<TFloat>(key + "SigmaMid");
        tSysPre_  = parameters->Get<TFloat>(key + "TSysPre");
        tDiasPre_ = parameters->Get<TFloat>(key + "TDiasPre");
        gamma_    = parameters->Get<TFloat>(key + "Gamma");
    }
    
protected:
    TFloat dpdt(TFloat t, TFloat p) const {
        const TFloat gPre = 0.5 * (1.0 - std::tanh((t - tDiasPre_) / gamma_));
        const TFloat fPre = 0.5 * (1.0 + std::tanh((t - tSysPre_) / gamma_)) * gPre;
        const TFloat b = alphaMax_ * fPre + alphaMin_ * (1.0 - fPre) + alphaPre_ * gPre + alphaMid_;
        return -std::abs(b) * p + sigmaMid_ * std::max(b, 0.0) + sigmaPre_ * std::max(gPre, 0.0);
    }
    
    /// resolves the switches of width gamma, 5 ms in the benchmark, to the accuracy of a converged solve
    static constexpr TFloat maxSubStep_ = 1e-4;
    TFloat alphaMin_;
    TFloat alphaMax_;
    TFloat alphaPre_;
    TFloat alphaMid_;
    TFloat sigmaPre_;
    TFloat sigmaMid_;
    TFloat tSysPre_;
    TFloat tDiasPre_;
    TFloat gamma_;
};
