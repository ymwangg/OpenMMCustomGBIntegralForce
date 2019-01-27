#ifndef _CustomGBIntegral_H_
#define _CustomGBIntegral_H_

#include "openmm/System.h"
#include "CudaContext.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/CharmmGBMVForce.h"
#include "CudaParameterSet.h"
#include <cuda.h>
namespace OpenMM{
class CustomGBIntegral{
    public:
        CustomGBIntegral(CudaContext& cu, const System& system, const CharmmGBMVForce& force, CudaParameterSet* &computedIntegrals, CudaParameterSet* &energyDerivs);
        ~CustomGBIntegral();
        void BeforeComputation(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& posData){
            return;
        }
        void FinishComputation(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& posData){
            return;
        }
        void initialize(const OpenMM::System& system, const OpenMM::CharmmGBMVForce& force){
            return;
        }
        void reduce();
        void evaluate();
        void computeLookupTable();
        void setLookupTableGridLength(double length);
        void setLookupTableBufferLength(double length);

    protected:
        std::vector<double> _atomicRadii;
    private:
        CudaParameterSet* radius;
        CudaParameterSet* computedIntegrals;
        CudaParameterSet* dEdI;
        CudaContext& cu;
        const System& system;
        const CharmmGBMVForce& force;
        CudaArray d_lookupTable;
        CudaArray d_lookupTableNumAtoms;
        CudaArray d_lookupTableMinCoord;
        CudaArray d_lookupTableGridStep;
        CudaArray d_lookupTableNumGridPoints;
        CudaArray d_quad;
        CudaArray d_volume;
        std::vector<CudaArray> d_quad_w;
        float _r0;
        float _r1;
        std::vector<void*> lookupTableArgs, integralArgs, reduceForceArgs;
        int _lookupTableSize;
        int _numIntegrals;
        int _numQuadPoints;
        std::vector<int> _integralOrders;
        float _lookupTableMinCoordinate[3];
        float _lookupTableGridLength;
        float _lookupTableBufferLength;
        float _lookupTableGridStep[3];
        int _lookupTableNumberOfGridPoints[3];
        bool _periodic;
        OpenMM::Vec3 _periodicBoxVectors[3];
        CUfunction lookupTableKernel,integralKernel,reduceForceKernel;
        int counter;
};
}

#endif
