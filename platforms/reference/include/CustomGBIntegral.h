#if false
#define aw
#ifndef _CustomGBIntegral_H_
#define _CustomGBIntegral_H_

#include "openmm/System.h"
#include "openmm/CharmmGBMVForce.h"
namespace OpenMM{
class CustomGBIntegral{
    public:
        virtual ~CustomGBIntegral(){
        };
        virtual void initialize(const OpenMM::System& system, const CharmmGBMVForce& force);
        virtual void evaluate(const int atomI, OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& atomPositions, const std::vector<int>& orders, std::vector<double>& values, std::vector<OpenMM::Vec3>& gradients);
};
}

#endif
#endif
