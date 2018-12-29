#ifndef __CharmmReferenceGBSW_H__
#define __CharmmReferenceGBSW_H__
namespace OpenMM{
class CharmmReferenceGBSW{
    public:
        CharmmReferenceGBSW(int numberOfAtoms);
        ~CharmmReferenceGBSW();
        double computeEnergyForces(const std::vector<OpenMM::Vec3>& atomCoordinates,
                const std::vector<double>& partialCharges, std::vector<OpenMM::Vec3>& forces);
        void computeBornRadii(const std::vector<OpenMM::Vec3>& atomCoordinates,
                const std::vector<double>& partialCharges, std::vector<double>& bornRadii);
        double computeVolume(const std::vector<OpenMM::Vec3>& atomCoordinates, const OpenMM::Vec3& quadCoordinate, const double switchDistance);
        void compute_dbornR_dr_vec(const std::vector<OpenMM::Vec3>& atomCoordinates, const int atomI, const double prefactor, const OpenMM::Vec3& quadCoordinate, const double volumeI, const double switchDistance);

        void setAtomicRadii(const std::vector<double>& atomicRadii);
        const std::vector<double>& getAtomicRadii() const;

        void setScaledRadiusFactors(const std::vector<double>& scaledRadiusFactors);
        const std::vector<double>& getScaledRadiusFactors() const;

        void setSolventDielectric(double solventDielectric);
        double getSolventDielectric() const;

        void setSoluteDielectric(double soluteDielectric);
        double getSoluteDielectric() const;

        void setUseCutoff(double distance);
        bool getUseCutoff() const;
        double getCutoffDistance() const;

        void setPeriodic(OpenMM::Vec3* vectors);
        bool getPeriodic();
        const OpenMM::Vec3* getPeriodicBox();

    private:
        std::vector<double> _atomicRadii;
        std::vector<double> _scaledRadiusFactors;
        std::vector<std::vector<OpenMM::Vec3 > > _dbornR_dr_vec;
        std::vector<double> _dG_dbornR;
        int _numberOfAtoms;
        double _solventDielectric;
        double _soluteDielectric;
        double _cutoffDistance;
        double _electricConstant;
        OpenMM::Vec3 _periodicBoxVectors[3];
        bool _cutoff;
        bool _periodic;
};
}
#endif
