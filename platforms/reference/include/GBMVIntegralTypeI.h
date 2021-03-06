#ifndef _GBMVIntegralTypeI_H_
#define _GBMVIntegralTypeI_H_

#include "CustomGBIntegral.h"
#include "openmm/System.h"
#include "openmm/CharmmGBMVForce.h"
#include "openmm/internal/ContextImpl.h"
namespace OpenMM{
class GBMVIntegralTypeI : public CustomGBIntegral {
    public:
        GBMVIntegralTypeI();
        ~GBMVIntegralTypeI(){
        }
        void initialize(const OpenMM::System& system, const CharmmGBMVForce& force);
        void evaluate(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<double>& values, std::vector<double>& gradients, const bool includeGradient = true);
        void BeforeComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates);
        void FinishComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates);
    private:
        void setBoxVectors(OpenMM::Vec3* vectors);
        double computeVolumeFromLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const std::vector<int>& atomList, const int numListAtoms);
        void computeGradientPerQuadFromLookupTable(const int atomI, const int valueIdx, const std::vector<OpenMM::Vec3>& atomCoordinates,const OpenMM::Vec3& r_q, const double V_q, std::vector<double>& gradients, const double prefactor, const std::vector<int>& atomList, const int numListAtoms);
        double computeVolume(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q);
        void computeGradientPerQuad(const int atomI, const int integralIdx, const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const double pre_sum, std::vector<double>& gradients, const double prefactor);

        int _numIntegrals;
        int _numParticles;
        std::vector<int> _orders;
        std::vector<std::vector<double> > _quad;
        double _r0;
        double _r1;
        double _gamma0;
        double _lambda;
        double _beta;
        bool _periodic;
        bool _useLookupTable;
        OpenMM::Vec3 _periodicBoxVectors[3];
};
}

#endif
