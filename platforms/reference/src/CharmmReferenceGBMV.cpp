// timing
#include <chrono>
#define BEGIN auto start = std::chrono::system_clock::now();
#define END auto end = std::chrono::system_clock::now();std::chrono::duration<double> elapsed_seconds = end-start;cout<<"calculateIxn elapsed time: " << elapsed_seconds.count()<<endl;

#include <string.h>
#include <sstream>
#include "CustomGBIntegral.h"
#include "SimTKOpenMMUtilities.h"
#include "ReferenceForce.h"
#include "CharmmReferenceGBMV.h"
#include <iostream>
using namespace::std;

using std::map;
using std::set;
using std::string;
using std::stringstream;
using std::vector;
using namespace OpenMM;

/**---------------------------------------------------------------------------------------

   ReferenceCustomGBIxn constructor

   --------------------------------------------------------------------------------------- */

CharmmReferenceGBMV::CharmmReferenceGBMV(const int numberOfAtoms, const std::vector<std::string>& integralNames, 
                     INTEGRAL& integral,
                     const vector<Lepton::CompiledExpression>& valueExpressions,
                     const vector<vector<Lepton::CompiledExpression> > valueDerivExpressions,
                     const vector<vector<Lepton::CompiledExpression> > valueGradientExpressions,
                     const vector<vector<Lepton::CompiledExpression> > valueParamDerivExpressions,
                     const vector<string>& valueNames,
                     const vector<OpenMM::CharmmGBMVForce::ComputationType>& valueTypes,
                     const vector<Lepton::CompiledExpression>& energyExpressions,
                     const vector<vector<Lepton::CompiledExpression> > energyDerivExpressions,
                     const vector<vector<Lepton::CompiledExpression> > energyGradientExpressions,
                     const vector<vector<Lepton::CompiledExpression> > energyParamDerivExpressions,
                     const vector<OpenMM::CharmmGBMVForce::ComputationType>& energyTypes,
                     const vector<string>& parameterNames) :
            numberOfAtoms(numberOfAtoms), cutoff(false), periodic(false), valueExpressions(valueExpressions), valueDerivExpressions(valueDerivExpressions), valueGradientExpressions(valueGradientExpressions), valueParamDerivExpressions(valueParamDerivExpressions),
            valueTypes(valueTypes), energyExpressions(energyExpressions), energyDerivExpressions(energyDerivExpressions), energyGradientExpressions(energyGradientExpressions), energyParamDerivExpressions(energyParamDerivExpressions),
            energyTypes(energyTypes)  {
    integralMethod = &integral;
    numberOfIntegrals = integralNames.size();
    numberOfValues = valueNames.size();
    volumeIntegralGradients.resize(numberOfIntegrals*numberOfAtoms*numberOfAtoms*3); 
    integrals.resize(numberOfIntegrals*numberOfAtoms);

    for (int i = 0; i < this->valueExpressions.size(); i++)
        expressionSet.registerExpression(this->valueExpressions[i]);
    for (int i = 0; i < this->valueDerivExpressions.size(); i++)
        for (int j = 0; j < this->valueDerivExpressions[i].size(); j++)
            expressionSet.registerExpression(this->valueDerivExpressions[i][j]);
    for (int i = 0; i < this->valueGradientExpressions.size(); i++)
        for (int j = 0; j < this->valueGradientExpressions[i].size(); j++)
            expressionSet.registerExpression(this->valueGradientExpressions[i][j]);
    for (int i = 0; i < this->valueParamDerivExpressions.size(); i++)
        for (int j = 0; j < this->valueParamDerivExpressions[i].size(); j++)
            expressionSet.registerExpression(this->valueParamDerivExpressions[i][j]);
    for (int i = 0; i < this->energyExpressions.size(); i++)
        expressionSet.registerExpression(this->energyExpressions[i]);
    for (int i = 0; i < this->energyDerivExpressions.size(); i++)
        for (int j = 0; j < this->energyDerivExpressions[i].size(); j++)
            expressionSet.registerExpression(this->energyDerivExpressions[i][j]);
    for (int i = 0; i < this->energyGradientExpressions.size(); i++)
        for (int j = 0; j < this->energyGradientExpressions[i].size(); j++)
            expressionSet.registerExpression(this->energyGradientExpressions[i][j]);
    for (int i = 0; i < this->energyParamDerivExpressions.size(); i++)
        for (int j = 0; j < this->energyParamDerivExpressions[i].size(); j++)
            expressionSet.registerExpression(this->energyParamDerivExpressions[i][j]);
    rIndex = expressionSet.getVariableIndex("r");
    xIndex = expressionSet.getVariableIndex("x");
    yIndex = expressionSet.getVariableIndex("y");
    zIndex = expressionSet.getVariableIndex("z");
    for (auto& param : parameterNames) {
        paramIndex.push_back(expressionSet.getVariableIndex(param));
        for (int j = 1; j < 3; j++) {
            stringstream name;
            name << param << j;
            particleParamIndex.push_back(expressionSet.getVariableIndex(name.str()));
        }
    }
    for(auto& integral : integralNames){
        integralIndex.push_back(expressionSet.getVariableIndex(integral));
        for (int j = 1; j < 3; j++) {
            stringstream name;
            name << integral << j;
            particleIntegralIndex.push_back(expressionSet.getVariableIndex(name.str()));
        }
    }
    for (auto& value : valueNames) {
        valueIndex.push_back(expressionSet.getVariableIndex(value));
        for (int j = 1; j < 3; j++) {
            stringstream name;
            name << value << j;
            particleValueIndex.push_back(expressionSet.getVariableIndex(name.str()));
        }
    }
}

/**---------------------------------------------------------------------------------------

   CharmmReferenceGBMV destructor

   --------------------------------------------------------------------------------------- */

CharmmReferenceGBMV::~CharmmReferenceGBMV() {
}

  /**---------------------------------------------------------------------------------------

     Set the force to use a cutoff.

     @param distance            the cutoff distance
     @param neighbors           the neighbor list to use

     --------------------------------------------------------------------------------------- */

  void CharmmReferenceGBMV::setUseCutoff(double distance, const OpenMM::NeighborList& neighbors) {

    cutoff = true;
    cutoffDistance = distance;
    neighborList = &neighbors;
  }

  /**---------------------------------------------------------------------------------------

     Set the force to use periodic boundary conditions.  This requires that a cutoff has
     also been set, and the smallest side of the periodic box is at least twice the cutoff
     distance.

     @param vectors    the vectors defining the periodic box

     --------------------------------------------------------------------------------------- */

  void CharmmReferenceGBMV::setPeriodic(Vec3* vectors) {

    if (cutoff) {
        assert(vectors[0][0] >= 2.0*cutoffDistance);
        assert(vectors[1][1] >= 2.0*cutoffDistance);
        assert(vectors[2][2] >= 2.0*cutoffDistance);
    }
    periodic = true;
    periodicBoxVectors[0] = vectors[0];
    periodicBoxVectors[1] = vectors[1];
    periodicBoxVectors[2] = vectors[2];
  }

void CharmmReferenceGBMV::calculateIxn(vector<Vec3>& atomCoordinates, double** atomParameters,
        const vector<set<int> >& exclusions, map<string, double>& globalParameters, 
        vector<Vec3>& forces, double* totalEnergy, double* energyParamDerivs, ContextImpl& inContext) {
    context = &inContext;
    for (auto& param : globalParameters)
        expressionSet.setVariable(expressionSet.getVariableIndex(param.first), param.second);

    //compute volume integral and its gradients to all atoms
    if(numberOfIntegrals > 0){
        integralMethod->evaluate(inContext, atomCoordinates, integrals, volumeIntegralGradients, true);
    }
    /*
    for(int i=0; i<numberOfAtoms; i++){
        //printf("%f %f %f -> ",atomCoordinates[i][0],atomCoordinates[i][1],atomCoordinates[i][2]);
        //printf("%f ",integralMethod->_atomicRadii[i]);
        for(int j=0; j<numberOfIntegrals; j++){
            printf("%f ",integrals[j*numberOfAtoms+i]);
        }
        printf("\n");
    }
    */
    
    if(numberOfValues > 0){
        int numDerivs = valueParamDerivExpressions[0].size(); //number of parameter derivatives
        values.resize(numberOfValues,vector<double>(numberOfAtoms,0.0));
        dEdI.resize(numberOfIntegrals, vector<double>(numberOfAtoms, 0.0));
        dEdV.resize(numberOfValues, vector<double>(numberOfAtoms, 0.0));
        dValuedParam.resize(numberOfValues);
        for (int i = 0; i < numberOfValues; i++)
            dValuedParam[i].resize(numDerivs, vector<double>(numberOfAtoms, 0.0));
        // First calculate the computed values.
        for (int valueIndex = 0; valueIndex < numberOfValues; valueIndex++) {
            calculateSingleParticleValue(valueIndex, numberOfAtoms, atomCoordinates, atomParameters);
        }
    }
    
    for(int i=0; i<numberOfAtoms; i++){
        //printf("%f\n",values[0][i]*10);
    }

    // Now calculate the energy and its derivates.

    for (int termIndex = 0; termIndex < (int) energyExpressions.size(); termIndex++) {
        if (energyTypes[termIndex] == OpenMM::CharmmGBMVForce::SingleParticle)
            calculateSingleParticleEnergyTerm(termIndex, numberOfAtoms, atomCoordinates, atomParameters, forces, totalEnergy, energyParamDerivs);
        else if (energyTypes[termIndex] == OpenMM::CharmmGBMVForce::ParticlePair)
            calculateParticlePairEnergyTerm(termIndex, numberOfAtoms, atomCoordinates, atomParameters, exclusions, true, forces, totalEnergy, energyParamDerivs);
        else
            calculateParticlePairEnergyTerm(termIndex, numberOfAtoms, atomCoordinates, atomParameters, exclusions, false, forces, totalEnergy, energyParamDerivs);
    }

    // Apply the chain rule to evaluate forces.

    calculateChainRuleForces(numberOfAtoms, atomCoordinates, atomParameters, exclusions, forces, energyParamDerivs);
}

void CharmmReferenceGBMV::calculateSingleParticleValue(int index, int numAtoms, vector<Vec3>& atomCoordinates, double** atomParameters) {
    for (int i = 0; i < numAtoms; i++) {
        expressionSet.setVariable(xIndex, atomCoordinates[i][0]);
        expressionSet.setVariable(yIndex, atomCoordinates[i][1]);
        expressionSet.setVariable(zIndex, atomCoordinates[i][2]);
        for (int j = 0; j < (int) paramIndex.size(); j++)
            expressionSet.setVariable(paramIndex[j], atomParameters[i][j]);
        for (int j = 0; j < numberOfIntegrals; j++)
            expressionSet.setVariable(integralIndex[j], integrals[j*numberOfAtoms + i]);
        for (int j = 0; j < index; j++)
            expressionSet.setVariable(valueIndex[j], values[j][i]);
        values[index][i] = valueExpressions[index].evaluate();

        // Calculate derivatives with respect to parameters.

        for (int j = 0; j < valueParamDerivExpressions[index].size(); j++)
            dValuedParam[index][j][i] += valueParamDerivExpressions[index][j].evaluate();
        for (int j = 0; j < index; j++) {
            double dVdV = valueDerivExpressions[index][j].evaluate();
            for (int k = 0; k < valueParamDerivExpressions[index].size(); k++)
                dValuedParam[index][k][i] += dVdV*dValuedParam[j][k][i];
        }
    }
}

void CharmmReferenceGBMV::calculateSingleParticleEnergyTerm(int index, int numAtoms, vector<Vec3>& atomCoordinates,
        double** atomParameters, vector<Vec3>& forces, double* totalEnergy, double* energyParamDerivs) {
    for (int i = 0; i < numAtoms; i++) {
        expressionSet.setVariable(xIndex, atomCoordinates[i][0]);
        expressionSet.setVariable(yIndex, atomCoordinates[i][1]);
        expressionSet.setVariable(zIndex, atomCoordinates[i][2]);
        for (int j = 0; j < (int) paramIndex.size(); j++)
            expressionSet.setVariable(paramIndex[j], atomParameters[i][j]);
        for (int j = 0; j < numberOfIntegrals; j++)
            expressionSet.setVariable(integralIndex[j], integrals[j*numberOfAtoms + i]);
        for (int j = 0; j < numberOfValues; j++)
            expressionSet.setVariable(valueIndex[j], values[j][i]);
        
        // Compute energy and force.
        
        if (totalEnergy != NULL)
            *totalEnergy += energyExpressions[index].evaluate();
        for (int j = 0; j < numberOfIntegrals; j++){
            dEdI[j][i] += energyDerivExpressions[index][j].evaluate();
        }
        for (int j = 0; j < numberOfValues; j++){
            dEdV[j][i] += energyDerivExpressions[index][j+numberOfIntegrals].evaluate();
            //cout<<dEdV[j][i]<<" ";
        }
        //cout<<endl;
        forces[i][0] -= energyGradientExpressions[index][0].evaluate();
        forces[i][1] -= energyGradientExpressions[index][1].evaluate();
        forces[i][2] -= energyGradientExpressions[index][2].evaluate();
        
        // Compute derivatives with respect to parameters.
        
        for (int k = 0; k < energyParamDerivExpressions[index].size(); k++)
            energyParamDerivs[k] += energyParamDerivExpressions[index][k].evaluate();
    }
}

void CharmmReferenceGBMV::calculateParticlePairEnergyTerm(int index, int numAtoms, vector<Vec3>& atomCoordinates, double** atomParameters,
        const vector<set<int> >& exclusions, bool useExclusions, vector<Vec3>& forces, double* totalEnergy, double* energyParamDerivs) {
    if (cutoff) {
        // Loop over all pairs in the neighbor list.
        for (auto& pair : *neighborList) {
            if (useExclusions && exclusions[pair.first].find(pair.second) != exclusions[pair.first].end())
                continue;
            calculateOnePairEnergyTerm(index, pair.first, pair.second, atomCoordinates, atomParameters, forces, totalEnergy, energyParamDerivs);
        }
    }
    else {
        // Perform an O(N^2) loop over all atom pairs.
        for (int i = 0; i < numberOfAtoms; i++) {
            for (int j = i+1; j < numAtoms; j++) {
                if (useExclusions && exclusions[i].find(j) != exclusions[i].end())
                    continue;
                calculateOnePairEnergyTerm(index, i, j, atomCoordinates, atomParameters, forces, totalEnergy, energyParamDerivs);
           }
        }
    }
}

void CharmmReferenceGBMV::calculateOnePairEnergyTerm(int index, int atom1, int atom2, vector<Vec3>& atomCoordinates, double** atomParameters,
        vector<Vec3>& forces, double* totalEnergy, double* energyParamDerivs) {
    // Compute the displacement.

    double deltaR[ReferenceForce::LastDeltaRIndex];
    if (periodic)
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[atom2], atomCoordinates[atom1], periodicBoxVectors, deltaR);
    else
        ReferenceForce::getDeltaR(atomCoordinates[atom2], atomCoordinates[atom1], deltaR);
    double r = deltaR[ReferenceForce::RIndex];
    if (cutoff && r >= cutoffDistance)
        return;

    // Record variables for evaluating expressions.

    expressionSet.setVariable(rIndex, r);
    for (int i = 0; i < (int) paramIndex.size(); i++) {
        expressionSet.setVariable(particleParamIndex[i*2], atomParameters[atom1][i]);
        expressionSet.setVariable(particleParamIndex[i*2+1], atomParameters[atom2][i]);
    }
    //set integral values
    for (int i = 0; i < numberOfIntegrals; i++) {
        expressionSet.setVariable(particleIntegralIndex[i*2], integrals[i*numberOfAtoms + atom1]);
        expressionSet.setVariable(particleIntegralIndex[i*2+1], integrals[i*numberOfAtoms + atom2]);
    }
    //set computed values
    for (int i = 0; i < numberOfValues; i++) {
        expressionSet.setVariable(particleValueIndex[i*2], values[i][atom1]);
        expressionSet.setVariable(particleValueIndex[i*2+1], values[i][atom2]);
    }

    // Evaluate the energy and its derivatives.
    if (totalEnergy != NULL)
        *totalEnergy += energyExpressions[index].evaluate();
    //calculate dE/dr
    double dEdR = energyDerivExpressions[index][0].evaluate();
    //normalize r/|r|
    dEdR *= 1/r;
    for (int i = 0; i < 3; i++) {
       forces[atom1][i] -= dEdR*deltaR[i];
       forces[atom2][i] += dEdR*deltaR[i];
    }
    //calculate dE/dI
    for (int i = 0; i < numberOfIntegrals; i++) {
        dEdI[i][atom1] += energyDerivExpressions[index][i*2+1].evaluate();
        dEdI[i][atom2] += energyDerivExpressions[index][i*2+2].evaluate();
    }
    //calculate dE/dValue
    for (int i = 0; i < numberOfValues; i++) {
        dEdV[i][atom1] += energyDerivExpressions[index][(i+numberOfIntegrals)*2+1].evaluate();
        dEdV[i][atom2] += energyDerivExpressions[index][(i+numberOfIntegrals)*2+2].evaluate();
    }
        
    // Compute derivatives with respect to parameters.

    for (int i = 0; i < energyParamDerivExpressions[index].size(); i++)
        energyParamDerivs[i] += energyParamDerivExpressions[index][i].evaluate();
}

void CharmmReferenceGBMV::calculateChainRuleForces(int numAtoms, vector<Vec3>& atomCoordinates, double** atomParameters,
        const vector<set<int> >& exclusions, vector<Vec3>& forces, double* energyParamDerivs) {
    for(int i = 0; i < numberOfAtoms; i++){
        expressionSet.setVariable(xIndex, atomCoordinates[i][0]);
        expressionSet.setVariable(yIndex, atomCoordinates[i][1]);
        expressionSet.setVariable(zIndex, atomCoordinates[i][2]);
        for (int j = 0; j < (int) paramIndex.size(); j++)
            expressionSet.setVariable(paramIndex[j], atomParameters[i][j]);
        for (int j = 0; j < numberOfIntegrals; j++)
            expressionSet.setVariable(integralIndex[j], integrals[j*numberOfAtoms + i]);
        for (int j = 0; j < numberOfValues; j++)
            expressionSet.setVariable(valueIndex[j], values[j][i]);
        //calculate dEdV using chain rule
        for (int j = 0; j < numberOfValues; j++){
            for(int k = 0; k < j; k++){
                double dVdV = valueDerivExpressions[j][k+numberOfIntegrals].evaluate();
                dEdV[k][i] += dEdV[j][i]*dVdV;
            }
        }
        //calculate dEdI
        for (int j = 0; j < numberOfValues; j++){
            for(int k = 0; k < numberOfIntegrals; k++){
                double dVdI = valueDerivExpressions[j][k].evaluate();
                dEdI[k][i] += dEdV[j][i]*dVdI;
            }
        }
        //update forces
        for (int j = 0; j < numberOfAtoms; j++){
            for (int k = 0; k < numberOfIntegrals; k++){
                for (int l = 0; l < 3; l++){
                    int grad_idx = k*numberOfAtoms*numberOfAtoms*3 + i*numberOfAtoms*3 + j*3 + l;
                    forces[j][l] -= dEdI[k][i]*volumeIntegralGradients[grad_idx];
                }
            }
        }
    }
    /*
    if (cutoff) {
        // Loop over all pairs in the neighbor list.

        for (auto& pair : *neighborList) {
            bool isExcluded = (exclusions[pair.first].find(pair.second) != exclusions[pair.first].end());
            calculateOnePairChainRule(pair.first, pair.second, atomCoordinates, atomParameters, forces, isExcluded);
            calculateOnePairChainRule(pair.second, pair.first, atomCoordinates, atomParameters, forces, isExcluded);
        }
    }
    else {
        // Perform an O(N^2) loop over all atom pairs.

        for (int i = 0; i < numAtoms; i++) {
            for (int j = i+1; j < numAtoms; j++) {
                bool isExcluded = (exclusions[i].find(j) != exclusions[i].end());
                calculateOnePairChainRule(i, j, atomCoordinates, atomParameters, forces, isExcluded);
                calculateOnePairChainRule(j, i, atomCoordinates, atomParameters, forces, isExcluded);
           }
        }
    }
    */

    // Compute chain rule terms for computed values that depend explicitly on particle coordinates.

    for (int i = 0; i < numberOfAtoms; i++) {
        expressionSet.setVariable(xIndex, atomCoordinates[i][0]);
        expressionSet.setVariable(yIndex, atomCoordinates[i][1]);
        expressionSet.setVariable(zIndex, atomCoordinates[i][2]);
        vector<double> dVdX(numberOfValues, 0.0);
        vector<double> dVdY(numberOfValues, 0.0);
        vector<double> dVdZ(numberOfValues, 0.0);
        for (int j = 0; j < (int) paramIndex.size(); j++)
            expressionSet.setVariable(paramIndex[j], atomParameters[i][j]);
        for (int j = 0; j < numberOfIntegrals; j++)
            expressionSet.setVariable(integralIndex[j], integrals[j*numberOfAtoms + i]);
        for (int j = 0; j < numberOfValues; j++) {
            expressionSet.setVariable(valueIndex[j], values[j][i]);
            for (int k = 0; k < j; k++) {
                double dVdV = valueDerivExpressions[j][k+numberOfIntegrals].evaluate();
                dVdX[j] += dVdV*dVdX[k];
                dVdY[j] += dVdV*dVdY[k];
                dVdZ[j] += dVdV*dVdZ[k];
            }
            dVdX[j] += valueGradientExpressions[j][0].evaluate();
            dVdY[j] += valueGradientExpressions[j][1].evaluate();
            dVdZ[j] += valueGradientExpressions[j][2].evaluate();
            forces[i][0] -= dEdV[j][i]*dVdX[j];
            forces[i][1] -= dEdV[j][i]*dVdY[j];
            forces[i][2] -= dEdV[j][i]*dVdZ[j];
        }
    }
        
    // Compute chain rule terms for derivatives with respect to parameters.

    for (int i = 0; i < numAtoms; i++)
        for (int j = 0; j < (int) valueIndex.size(); j++)
            for (int k = 0; k < dValuedParam[j].size(); k++)
                energyParamDerivs[k] += dEdV[j][i]*dValuedParam[j][k][i];
}

