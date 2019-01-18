#ifndef _CustomGBIntegral_H_
#define _CustomGBIntegral_H_

#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/CharmmGBMVForce.h"
namespace OpenMM{
class CustomGBIntegral{
    public:
        CustomGBIntegral();
        ~CustomGBIntegral();
        virtual void BeforeComputation(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& posData) = 0;
        virtual void FinishComputation(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& posData) = 0;
        virtual void initialize(const OpenMM::System& system, const OpenMM::CharmmGBMVForce& force) = 0;
        virtual void evaluate(const int atomI, OpenMM::ContextImpl& context, 
                const std::vector<OpenMM::Vec3>& atomCoordinates, std::vector<double>& values, 
                std::vector<double>& gradients, const bool includeGradient) = 0;
        void computeLookupTable(const std::vector<OpenMM::Vec3>& atomCoordinates);
        void getLookupTableAtomList(OpenMM::Vec3 point, std::vector<int>& atomList);
        void setPeriodic(OpenMM::Vec3* vectors);
        void setLookupTableGridLength(double length);
        void setLookupTableBufferLength(double length);

    protected:
        std::vector<double> _atomicRadii;
    private:
        bool validateTransform(OpenMM::Vec3 newVec);
        std::vector<std::vector<int> > _lookupTable;
        int _lookupTableSize;
        std::vector<int> _lookupTableNumAtoms;
        double _minCoordinate[3];
        double _maxCoordinate[3];
        double _lookupTableMinCoordinate[3];
        double _lookupTableMaxCoordinate[3];
        double _lookupTableGridLength;
        double _lookupTableBufferLength;
        int _lookupTableNumberOfGridPoints[3];
        bool _periodic;
        OpenMM::Vec3 _periodicBoxVectors[3];
};
}

#endif
