#ifndef _CustomGBIntegral_H_
#define _CustomGBIntegral_H_

#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/CharmmGBMVForce.h"
class CustomGBIntegral{
    public:
        virtual ~CustomGBIntegral(){
        };
        virtual void initialize(const OpenMM::System& system, const OpenMM::CharmmGBMVForce& force);
        virtual void evaluate(const int atomI, OpenMM::ContextImpl& context, 
                const std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<double>& values, 
                std::vector<std::vector<OpenMM::Vec3> >& gradients, const bool includeGradient);
};

#endif
