#ifndef _CustomGBIntegral_H_
#define _CustomGBIntegral_H_

#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/CharmmGBMVForce.h"
namespace OpenMM{
class CustomGBIntegral{
    public:
        CustomGBIntegral(){
        }
        virtual ~CustomGBIntegral(){
        }
        virtual void BeforeComputation(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& posData) = 0;
        virtual void FinishComputation(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& posData) = 0;
        virtual void initialize(const OpenMM::System& system, const OpenMM::CharmmGBMVForce& force) = 0;
        virtual void evaluate(const int atomI, OpenMM::ContextImpl& context, 
                const std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<double>& values, 
                std::vector<std::vector<OpenMM::Vec3> >& gradients, const bool includeGradient) = 0;
};
}

#endif
