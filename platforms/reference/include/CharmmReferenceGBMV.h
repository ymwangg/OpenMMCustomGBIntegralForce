#ifndef __CharmmReferenceGBMV_H__
#define __CharmmReferenceGBMV_H__

//#define INTEGRAL GBMVIntegralTypeI
//#define INTEGRAL GBSWIntegral
#define INTEGRAL CustomGBIntegral

#include "CustomGBIntegral.h"
#include "ReferenceNeighborList.h"
#include "openmm/internal/CompiledExpressionSet.h"
#include "openmm/CharmmGBMVForce.h"
#include <map>
#include <set>
#include <vector>

namespace OpenMM {

class CharmmReferenceGBMV {

   private:

      ContextImpl* context;
      INTEGRAL* integralMethod;
      bool cutoff;
      bool periodic;
      const OpenMM::NeighborList* neighborList;
      OpenMM::Vec3 periodicBoxVectors[3];
      double cutoffDistance;
      CompiledExpressionSet expressionSet;
      std::vector<Lepton::CompiledExpression> valueExpressions;
      std::vector<std::vector<Lepton::CompiledExpression> > valueDerivExpressions;
      std::vector<std::vector<Lepton::CompiledExpression> > valueGradientExpressions;
      std::vector<std::vector<Lepton::CompiledExpression> > valueParamDerivExpressions;
      std::vector<OpenMM::CharmmGBMVForce::ComputationType> valueTypes;
      std::vector<Lepton::CompiledExpression> energyExpressions;
      std::vector<std::vector<Lepton::CompiledExpression> > energyDerivExpressions;
      std::vector<std::vector<Lepton::CompiledExpression> > energyGradientExpressions;
      std::vector<std::vector<Lepton::CompiledExpression> > energyParamDerivExpressions;
      std::vector<OpenMM::CharmmGBMVForce::ComputationType> energyTypes;
      std::vector<int> paramIndex;
      std::vector<int> valueIndex;
      std::vector<int> integralIndex;
      std::vector<int> particleParamIndex;
      std::vector<int> particleValueIndex;
      std::vector<int> particleIntegralIndex;
      int rIndex, xIndex, yIndex, zIndex;
      int numberOfIntegrals, numberOfValues, numberOfAtoms;
      std::vector<std::vector<double> > values, dEdV, dEdI;
      std::vector<std::vector<std::vector<double> > > dValuedParam;
      std::vector<double> integrals, volumeIntegralGradients;

      /**---------------------------------------------------------------------------------------

         Calculate a computed value of type SingleParticle

         @param index            the index of the value to compute
         @param numAtoms         number of atoms
         @param atomCoordinates  atom coordinates
         @param atomParameters   atomParameters[atomIndex][paramterIndex]

         --------------------------------------------------------------------------------------- */

      void calculateSingleParticleValue(int index, int numAtoms, std::vector<OpenMM::Vec3>& atomCoordinates, double** atomParameters);

      /**---------------------------------------------------------------------------------------

         Calculate a computed value that is based on particle pairs

         @param index            the index of the value to compute
         @param numAtoms         number of atoms
         @param atomCoordinates  atom coordinates
         @param atomParameters   atomParameters[atomIndex][paramterIndex]
         @param exclusions       exclusions[i] is the set of excluded indices for atom i
         @param useExclusions    specifies whether to use exclusions

         --------------------------------------------------------------------------------------- */

      void calculateParticlePairValue(int index, int numAtoms, std::vector<OpenMM::Vec3>& atomCoordinates, double** atomParameters,
                                      const std::vector<std::set<int> >& exclusions, bool useExclusions);

      /**---------------------------------------------------------------------------------------

         Evaluate a single atom pair as part of calculating a computed value

         @param index            the index of the value to compute
         @param atom1            the index of the first atom in the pair
         @param atom2            the index of the second atom in the pair
         @param atomCoordinates  atom coordinates
         @param atomParameters   atomParameters[atomIndex][paramterIndex]

         --------------------------------------------------------------------------------------- */

      void calculateOnePairValue(int index, int atom1, int atom2, std::vector<OpenMM::Vec3>& atomCoordinates, double** atomParameters);

      /**---------------------------------------------------------------------------------------

         Calculate an energy term of type SingleParticle

         @param index            the index of the value to compute
         @param numAtoms         number of atoms
         @param atomCoordinates  atom coordinates
         @param atomParameters   atomParameters[atomIndex][paramterIndex]
         @param forces           forces on atoms are added to this
         @param totalEnergy      the energy contribution is added to this

         --------------------------------------------------------------------------------------- */

      void calculateSingleParticleEnergyTerm(int index, int numAtoms, std::vector<OpenMM::Vec3>& atomCoordinates,
                        double** atomParameters, std::vector<OpenMM::Vec3>& forces, double* totalEnergy, double* energyParamDerivs);

      /**---------------------------------------------------------------------------------------

         Calculate an energy term that is based on particle pairs

         @param index            the index of the term to compute
         @param numAtoms         number of atoms
         @param atomCoordinates  atom coordinates
         @param atomParameters   atomParameters[atomIndex][paramterIndex]
         @param exclusions       exclusions[i] is the set of excluded indices for atom i
         @param useExclusions    specifies whether to use exclusions
         @param forces           forces on atoms are added to this
         @param totalEnergy      the energy contribution is added to this

         --------------------------------------------------------------------------------------- */

      void calculateParticlePairEnergyTerm(int index, int numAtoms, std::vector<OpenMM::Vec3>& atomCoordinates, double** atomParameters,
                                      const std::vector<std::set<int> >& exclusions, bool useExclusions,
                                      std::vector<OpenMM::Vec3>& forces, double* totalEnergy, double* energyParamDerivs);

      /**---------------------------------------------------------------------------------------

         Evaluate a single atom pair as part of calculating an energy term

         @param index            the index of the term to compute
         @param atom1            the index of the first atom in the pair
         @param atom2            the index of the second atom in the pair
         @param atomCoordinates  atom coordinates
         @param atomParameters   atomParameters[atomIndex][paramterIndex]
         @param forces           forces on atoms are added to this
         @param totalEnergy      the energy contribution is added to this

         --------------------------------------------------------------------------------------- */

      void calculateOnePairEnergyTerm(int index, int atom1, int atom2, std::vector<OpenMM::Vec3>& atomCoordinates, double** atomParameters,
                                 std::vector<OpenMM::Vec3>& forces, double* totalEnergy, double* energyParamDerivs);

      /**---------------------------------------------------------------------------------------

         Apply the chain rule to compute forces on atoms

         @param numAtoms         number of atoms
         @param atomCoordinates  atom coordinates
         @param atomParameters   atomParameters[atomIndex][paramterIndex]
         @param exclusions       exclusions[i] is the set of excluded indices for atom i
         @param forces           forces on atoms are added to this

         --------------------------------------------------------------------------------------- */

      void calculateChainRuleForces(int numAtoms, std::vector<OpenMM::Vec3>& atomCoordinates, double** atomParameters,
                                      const std::vector<std::set<int> >& exclusions, std::vector<OpenMM::Vec3>& forces, double* energyParamDerivs);

      /**---------------------------------------------------------------------------------------

         Evaluate a single atom pair as part of applying the chain rule

         @param atom1            the index of the first atom in the pair
         @param atom2            the index of the second atom in the pair
         @param atomCoordinates  atom coordinates
         @param atomParameters   atomParameters[atomIndex][paramterIndex]
         @param forces           forces on atoms are added to this
         @param isExcluded       specifies whether this is an excluded pair

         --------------------------------------------------------------------------------------- */

      void calculateOnePairChainRule(int atom1, int atom2, std::vector<OpenMM::Vec3>& atomCoordinates, double** atomParameters,
                                 std::vector<OpenMM::Vec3>& forces, bool isExcluded);

   public:

      /**---------------------------------------------------------------------------------------

         Constructor

         --------------------------------------------------------------------------------------- */

       CharmmReferenceGBMV(const int numberOfAtoms, const std::vector<std::string>& integralNames, INTEGRAL& integral,
                            const std::vector<Lepton::CompiledExpression>& valueExpressions,
                            const std::vector<std::vector<Lepton::CompiledExpression> > valueDerivExpressions,
                            const std::vector<std::vector<Lepton::CompiledExpression> > valueGradientExpressions,
                            const std::vector<std::vector<Lepton::CompiledExpression> > valueParamDerivExpressions,
                            const std::vector<std::string>& valueNames,
                            const std::vector<OpenMM::CharmmGBMVForce::ComputationType>& valueTypes,
                            const std::vector<Lepton::CompiledExpression>& energyExpressions,
                            const std::vector<std::vector<Lepton::CompiledExpression> > energyDerivExpressions,
                            const std::vector<std::vector<Lepton::CompiledExpression> > energyGradientExpressions,
                            const std::vector<std::vector<Lepton::CompiledExpression> > energyParamDerivExpressions,
                            const std::vector<OpenMM::CharmmGBMVForce::ComputationType>& energyTypes,
                            const std::vector<std::string>& parameterNames);

      /**---------------------------------------------------------------------------------------

         Destructor

         --------------------------------------------------------------------------------------- */

       ~CharmmReferenceGBMV();

      /**---------------------------------------------------------------------------------------

         Set the force to use a cutoff.

         @param distance            the cutoff distance
         @param neighbors           the neighbor list to use

         --------------------------------------------------------------------------------------- */

      void setUseCutoff(double distance, const OpenMM::NeighborList& neighbors);

      /**---------------------------------------------------------------------------------------

         Set the force to use periodic boundary conditions.  This requires that a cutoff has
         already been set, and the smallest side of the periodic box is at least twice the cutoff
         distance.

         @param vectors    the vectors defining the periodic box

         --------------------------------------------------------------------------------------- */

      void setPeriodic(OpenMM::Vec3* vectors);

      /**---------------------------------------------------------------------------------------

         Calculate custom GB ixn

         @param numberOfAtoms    number of atoms
         @param atomCoordinates  atom coordinates
         @param atomParameters   atomParameters[atomIndex][paramterIndex]
         @param exclusions       exclusions[i] is the set of excluded indices for atom i
         @param globalParameters the values of global parameters
         @param forces           force array (forces added)
         @param totalEnergy      total energy

         --------------------------------------------------------------------------------------- */

      void calculateIxn(std::vector<OpenMM::Vec3>& atomCoordinates, double** atomParameters, const std::vector<std::set<int> >& exclusions, std::map<std::string, double>& globalParameters, std::vector<OpenMM::Vec3>& forces, double* totalEnergy, double* energyParamDerivs, ContextImpl& inContext);

// ---------------------------------------------------------------------------------------

};

} // namespace OpenMM

#endif // __CharmmReferenceGBMV_H_H
