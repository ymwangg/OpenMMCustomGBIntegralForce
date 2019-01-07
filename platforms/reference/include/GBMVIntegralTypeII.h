#ifndef _GBMVIntegralTypeII_H_
#define _GBMVIntegralTypeII_H_

#include "CustomGBIntegral.h"
#include "openmm/System.h"
#include "openmm/CharmmGBMVForce.h"
#include "openmm/internal/ContextImpl.h"
namespace OpenMM{
class GBMVIntegralTypeII : public CustomGBIntegral {
    public:
        GBMVIntegralTypeII();
        ~GBMVIntegralTypeII(){
        }
        void initialize(const OpenMM::System& system, const CharmmGBMVForce& force);
        void evaluate(const int atomI, OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<double>& values, std::vector<std::vector<OpenMM::Vec3> >& gradients, const bool includeGradient = true);
        void BeforeComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates);
        void FinishComputation(ContextImpl& context, const std::vector<OpenMM::Vec3>& atomCoordinates);
    private:
        void setBoxVectors(OpenMM::Vec3* vectors);
        void computeLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates);
        std::vector<int> getLookupTableAtomList(OpenMM::Vec3 point);
        double computeVolumeFromLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& r_q, const std::vector<int>& atomList, double& sum1, double& sum2, double& sum3, OpenMM::Vec3& denom_vec);
        void computeGradientPerQuadFromLookupTable(const int atomI, const std::vector<OpenMM::Vec3>& atomCoordinates,const OpenMM::Vec3& r_q, const double V_q, std::vector<OpenMM::Vec3>& gradients, const double prefactor, const std::vector<int>& atomList, const double sum1, const double sum2, const double sum3, const OpenMM::Vec3& denom_vec);

        int _numIntegrals;
        int _numParticles;
        std::vector<int> _orders;
        std::vector<std::vector<double> > _quad;
        std::vector<double> _atomicRadii;
        bool _periodic;
        double _r0;
        double _r1;
        double _gamma0;
        double _lambda;
        double _beta;
        double _P1;
        double _P2;
        double _S0;

        OpenMM::Vec3 _periodicBoxVectors[3];

        std::vector<std::vector<int>> _lookupTable;
        double _lookupTableMinCoordinate[3];
        double _lookupTableMaxCoordinate[3];
        double _lookupTableGridLength;
        double _lookupTableBufferLength;
        int _lookupTableNumberOfGridPoints[3];
};
}

#endif
