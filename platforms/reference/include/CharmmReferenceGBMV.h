#ifndef __CharmmReferenceGBMV_H__
#define __CharmmReferenceGBMV_H__
namespace OpenMM{
class CharmmReferenceGBMV{
    public:
        CharmmReferenceGBMV(int numberOfAtoms);
        ~CharmmReferenceGBMV();
        double computeEnergyForces(const std::vector<OpenMM::Vec3>& atomCoordinates,
                const std::vector<double>& partialCharges, std::vector<OpenMM::Vec3>& forces);
        void computeBornRadii(const std::vector<OpenMM::Vec3>& atomCoordinates,
                const std::vector<double>& partialCharges, std::vector<double>& bornRadii);
        double computeVolume(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& quadCoordinate);
        void compute_dbornR_dr_vec(const std::vector<OpenMM::Vec3>& atomCoordinates, const int atomI, const double prefactor, const OpenMM::Vec3& quadCoordinate, const double volumeI);
        void computeLookupTable(const std::vector<Vec3>& atomCoordinates);
        void computeBornRadiiFast(const std::vector<OpenMM::Vec3>& atomCoordinates,
                const std::vector<double>& partialCharges, std::vector<double>& bornRadii);
        double computeVolumeFast(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& quadCoordinate, const double switchDistance, const std::vector<int>& atomList);
        void compute_dbornR_dr_vec_Fast(const std::vector<OpenMM::Vec3>& atomCoordinates, const int atomI, const double prefactor, const OpenMM::Vec3& quadCoordinate, const double volumeI, const double switchDistance, const std::vector<int>& atomList);
        std::vector<int> getLookupTableAtomList(OpenMM::Vec3 point);

        void setAtomicRadii(const std::vector<double>& atomicRadii);
        const std::vector<double>& getAtomicRadii() const;

        void setScaledRadiusFactors(const std::vector<double>& scaledRadiusFactors);
        const std::vector<double>& getScaledRadiusFactors() const;

        void setSolventDielectric(double solventDielectric);
        double getSolventDielectric() const;

        void setSoluteDielectric(double soluteDielectric);
        double getSoluteDielectric() const;

        void setUseCutoff(double distance);
        void setNeighborList(OpenMM::NeighborList& neighborList);
        OpenMM::NeighborList* getNeighborList();
        bool getUseCutoff() const;
        double getCutoffDistance() const;

        void setNoCutoff();
        void setNoPeriodic();
        void setPeriodic();

        void setPeriodic(OpenMM::Vec3* vectors);
        bool getPeriodic();
        const OpenMM::Vec3* getPeriodicBox();

    private:
        OpenMM::NeighborList* _neighborList;
        std::vector<double> _atomicRadii;
        std::vector<double> _scaledRadiusFactors;
        std::vector<std::vector<OpenMM::Vec3 > > _dbornR_dr_vec;
        std::vector<double> _dG_dbornR;
        std::vector<std::vector<double> > _quad;
        std::vector<std::vector<int>> _lookupTable;
        int _numberOfAtoms;
        double _solventDielectric;
        double _soluteDielectric;
        double _cutoffDistance;
        double _electricConstant;
        double _r0;
        double _r1;
        OpenMM::Vec3 _periodicBoxVectors[3];
        bool _cutoff;
        bool _periodic;
        double _lookupTableMinCoordinate[3];
        double _lookupTableMaxCoordinate[3];
        double _lookupTableGridLength;
        double _lookupTableBufferLength;
        double _switchingDistance;
        int _lookupTableNumberOfGridPoints[3];
};
}
#endif
