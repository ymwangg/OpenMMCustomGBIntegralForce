#ifndef _GBSWIntegral_H_
#define _GBSWIntegral_H_

#include "openmm/System.h"
#include "openmm/CharmmGBMVForce.h"
#include "openmm/internal/ContextImpl.h"
namespace OpenMM{
class GBSWIntegral{
    public:
        GBSWIntegral();
        void initialize(const OpenMM::System& system, const CharmmGBMVForce& force);
        void evaluate(const int atomI, OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& atomPositions, const std::vector<int>& orders, std::vector<double>& values, std::vector<std::vector<OpenMM::Vec3> >& gradients, const bool includeGradient = true);
    private:
        std::vector<std::vector<double> > _quad;
        std::vector<double> _atomicRadii;
        void setBoxVectors(OpenMM::Vec3* vectors);
        double computeVolume(const std::vector<OpenMM::Vec3>& atomPositions, const OpenMM::Vec3& r_q);
        void computeGradientPerQuad(const int atomI, const std::vector<OpenMM::Vec3>& atomPositions, 
                const OpenMM::Vec3& r_q, const double V_q, std::vector<OpenMM::Vec3>& gradients, const double prefactor);
        double _r0;
        double _r1;
        double _sw;
        bool _periodic;
        OpenMM::Vec3 _periodicBoxVectors[3];
};
}

#endif
