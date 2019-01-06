/* -------------------------------------------------------------------------- *
 *                               OpenMMAmoeba                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "CharmmReferenceKernels.h"
#include "CharmmReferenceGBMV.h"
#include "CharmmReferenceGBSW.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "SimTKOpenMMUtilities.h"
#include "GBSWIntegral.h"
#include "ReferenceTabulatedFunction.h"
#include "lepton/CustomFunction.h"
#include "lepton/Operation.h"
#include "lepton/Parser.h"
#include "lepton/ParsedExpression.h"
#include <iostream>

using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->velocities);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *(Vec3*) data->periodicBoxSize;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

static map<string, double>& extractEnergyParameterDerivatives(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((map<string, double>*) data->energyParameterDerivatives);
}

static int** allocateIntArray(int length, int width) {
    int** array = new int*[length];
    for (int i = 0; i < length; ++i)
        array[i] = new int[width];
    return array;
}

static double** allocateRealArray(int length, int width) {
    double** array = new double*[length];
    for (int i = 0; i < length; ++i) 
        array[i] = new double[width];
    return array;
}

static void disposeIntArray(int** array, int size) {
    if (array) {
        for (int i = 0; i < size; ++i)
            delete[] array[i];
        delete[] array;
    }
}

static void disposeRealArray(double** array, int size) {
    if (array) {
        for (int i = 0; i < size; ++i) 
            delete[] array[i];
        delete[] array;
    }    
}
/**
 * Make sure an expression doesn't use any undefined variables.
 */
static void validateVariables(const Lepton::ExpressionTreeNode& node, const set<string>& variables) {
    const Lepton::Operation& op = node.getOperation();
    if (op.getId() == Lepton::Operation::VARIABLE && variables.find(op.getName()) == variables.end())
        throw OpenMMException("Unknown variable in expression: "+op.getName());
    for (auto& child : node.getChildren())
        validateVariables(child, variables);
}

// ***************************************************************************

ReferenceCalcCharmmGBMVForceKernel::~ReferenceCalcCharmmGBMVForceKernel() {
    disposeRealArray(particleParamArray, numParticles);
    if (neighborList != NULL)
        delete neighborList;
}

void ReferenceCalcCharmmGBMVForceKernel::initialize(const System& system, const CharmmGBMVForce& force) {
    integralMethod = new GBSWIntegral();
    if(force.getNumGBIntegrals()>0)
        integralMethod->initialize(system,force);
    /*
    if (force.getNumComputedValues() > 0) {
        string name, expression;
        CharmmGBMVForce::ComputationType type;
        force.getComputedValueParameters(0, name, expression, type);
        if (type == CharmmGBMVForce::SingleParticle)
            throw OpenMMException("ReferencePlatform requires that the first computed value for a CharmmGBMVForce be of type GBIntegral.");
        for (int i = 1; i < force.getNumComputedValues(); i++) {
            force.getComputedValueParameters(i, name, expression, type);
            if (type != CharmmGBMVForce::SingleParticle)
                throw OpenMMException("ReferencePlatform requires that a CharmmGBMVForce only have one computed value of type GBIntegral.");
        }
    }
    */

    // Record the exclusions.

    numParticles = force.getNumParticles();
    exclusions.resize(numParticles);
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
    }

   // Build the arrays.

    // per particle parameters
    int numPerParticleParameters = force.getNumPerParticleParameters();
    particleParamArray = allocateRealArray(numParticles, numPerParticleParameters);
    for (int i = 0; i < numParticles; ++i) {
        vector<double> parameters;
        force.getParticleParameters(i, parameters);
        for (int j = 0; j < numPerParticleParameters; j++)
            particleParamArray[i][j] = parameters[j];
    }
    for (int i = 0; i < numPerParticleParameters; i++)
        particleParameterNames.push_back(force.getPerParticleParameterName(i));
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalParameterNames.push_back(force.getGlobalParameterName(i));


    // nonbonded method
    nonbondedMethod = CalcCharmmGBMVForceKernel::NonbondedMethod(force.getNonbondedMethod());
    nonbondedCutoff = force.getCutoffDistance();
    if (nonbondedMethod == NoCutoff)
        neighborList = NULL;
    else
        neighborList = new NeighborList();

    // Create custom functions for the tabulated functions.

    map<string, Lepton::CustomFunction*> functions;
    for (int i = 0; i < force.getNumFunctions(); i++)
        functions[force.getTabulatedFunctionName(i)] = createReferenceTabulatedFunction(force.getTabulatedFunction(i));

    // Parse the expressions for computed values.

    valueDerivExpressions.resize(force.getNumComputedValues()+force.getNumGBIntegrals());
    valueGradientExpressions.resize(force.getNumComputedValues());
    valueParamDerivExpressions.resize(force.getNumComputedValues());
    set<string> particleVariables, pairVariables;
    pairVariables.insert("r");
    particleVariables.insert("x");
    particleVariables.insert("y");
    particleVariables.insert("z");
    for (int i = 0; i < numPerParticleParameters; i++) {
        particleVariables.insert(particleParameterNames[i]);
        pairVariables.insert(particleParameterNames[i]+"1");
        pairVariables.insert(particleParameterNames[i]+"2");
    }
    particleVariables.insert(globalParameterNames.begin(), globalParameterNames.end());
    pairVariables.insert(globalParameterNames.begin(), globalParameterNames.end());

    //add volume integral names
    for(int i = 0; i < force.getNumGBIntegrals(); i++){
        string name;
        force.getGBIntegralParameters(i, name);
        integralNames.push_back(name);
        particleVariables.insert(name);
        pairVariables.insert(name+"1");
        pairVariables.insert(name+"2");
    }

    for (int i = 0; i < force.getNumComputedValues(); i++) {
        string name, expression;
        CharmmGBMVForce::ComputationType type;
        force.getComputedValueParameters(i, name, expression, type);
        Lepton::ParsedExpression ex = Lepton::Parser::parse(expression, functions).optimize();
        valueExpressions.push_back(ex.createCompiledExpression());
        valueTypes.push_back(type);
        valueNames.push_back(name);
        valueGradientExpressions[i].push_back(ex.differentiate("x").createCompiledExpression());
        valueGradientExpressions[i].push_back(ex.differentiate("y").createCompiledExpression());
        valueGradientExpressions[i].push_back(ex.differentiate("z").createCompiledExpression());
        for (int j = 0; j < force.getNumGBIntegrals(); j++){
            valueDerivExpressions[i].push_back(ex.differentiate(integralNames[j]).createCompiledExpression());
        }
        for (int j = 0; j < i; j++)
            valueDerivExpressions[i].push_back(ex.differentiate(valueNames[j]).createCompiledExpression());
        validateVariables(ex.getRootNode(), particleVariables);
        for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++) {
            string param = force.getEnergyParameterDerivativeName(j);
            energyParamDerivNames.push_back(param);
            valueParamDerivExpressions[i].push_back(ex.differentiate(param).createCompiledExpression());
        }
        particleVariables.insert(name);
        pairVariables.insert(name+"1");
        pairVariables.insert(name+"2");
    }

    // Parse the expressions for energy terms.

    energyDerivExpressions.resize(force.getNumEnergyTerms()+force.getNumGBIntegrals());
    energyGradientExpressions.resize(force.getNumEnergyTerms());
    energyParamDerivExpressions.resize(force.getNumEnergyTerms());
    for (int i = 0; i < force.getNumEnergyTerms(); i++) {
        string expression;
        CharmmGBMVForce::ComputationType type;
        force.getEnergyTermParameters(i, expression, type);
        Lepton::ParsedExpression ex = Lepton::Parser::parse(expression, functions).optimize();
        energyExpressions.push_back(ex.createCompiledExpression());
        energyTypes.push_back(type);
        if (type != CharmmGBMVForce::SingleParticle)
            energyDerivExpressions[i].push_back(ex.differentiate("r").createCompiledExpression());
        for (int j = 0; j < force.getNumGBIntegrals(); j++){
            if (type == CharmmGBMVForce::SingleParticle) {
                energyDerivExpressions[i].push_back(ex.differentiate(integralNames[j]).createCompiledExpression());
                energyGradientExpressions[i].push_back(ex.differentiate("x").createCompiledExpression());
                energyGradientExpressions[i].push_back(ex.differentiate("y").createCompiledExpression());
                energyGradientExpressions[i].push_back(ex.differentiate("z").createCompiledExpression());
                validateVariables(ex.getRootNode(), particleVariables);
            }    
            else {
                energyDerivExpressions[i].push_back(ex.differentiate(integralNames[j]+"1").createCompiledExpression());
                energyDerivExpressions[i].push_back(ex.differentiate(integralNames[j]+"2").createCompiledExpression());
                validateVariables(ex.getRootNode(), pairVariables);
            }    

        }
        for (int j = 0; j < force.getNumComputedValues(); j++) {
            if (type == CharmmGBMVForce::SingleParticle) {
                energyDerivExpressions[i].push_back(ex.differentiate(valueNames[j]).createCompiledExpression());
                energyGradientExpressions[i].push_back(ex.differentiate("x").createCompiledExpression());
                energyGradientExpressions[i].push_back(ex.differentiate("y").createCompiledExpression());
                energyGradientExpressions[i].push_back(ex.differentiate("z").createCompiledExpression());
                validateVariables(ex.getRootNode(), particleVariables);
            }    
            else {
                energyDerivExpressions[i].push_back(ex.differentiate(valueNames[j]+"1").createCompiledExpression());
                energyDerivExpressions[i].push_back(ex.differentiate(valueNames[j]+"2").createCompiledExpression());
                validateVariables(ex.getRootNode(), pairVariables);
            }    
        }    
        for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++) 
            energyParamDerivExpressions[i].push_back(ex.differentiate(force.getEnergyParameterDerivativeName(j)).createCompiledExpression());
    }   
    // Delete the custom functions.

    for (auto& function : functions)
        delete function.second;
}

double ReferenceCalcCharmmGBMVForceKernel::validateIntegral(ContextImpl& context){
    int atomId = 0;
    vector<Vec3> posData = extractPositions(context);
    std::vector<std::vector<OpenMM::Vec3> > gradients;
    std::vector<std::vector<OpenMM::Vec3> > gradientsFD;
    std::vector<int> orders = {4};
    gradientsFD.resize(orders.size(),std::vector<OpenMM::Vec3>(posData.size()));
    double d=1e-6;
    std::vector<double> values1;
    integralMethod->evaluate(atomId, context, posData, values1, gradients, true);
    for(int i=0; i<posData.size(); ++i){
        for(int k=0; k<3; ++k){
            std::vector<double> values2;
            posData = extractPositions(context);
            posData[i][k] += d;
            integralMethod->evaluate(atomId, context, posData, values2, gradients, false);
            for(int j=0; j<values1.size(); ++j){
                gradientsFD[j][i][k] = (values2[j]-values1[j])/d;
            }
        }
    }
    double error = 0.0;
    int n=0;
    OpenMM::Vec3 delta;
    for(int i=0; i<orders.size(); ++i){
        for(int j=0; j<posData.size(); ++j){
            cout<<gradients[i][j]<<" "<<gradientsFD[i][j]<<endl;
            delta = gradients[i][j] - gradientsFD[i][j];
            error += delta.dot(delta);
            n++;
        }
    }
    error = sqrt(error/n); //should be below 0.1
    cout<<error<<endl;
    return error;
}
double ReferenceCalcCharmmGBMVForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    //validateIntegral(context);

    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);

    double energy = 0;
    int numIntegrals = integralNames.size();
    CharmmReferenceGBMV ixn(numParticles, integralNames, *integralMethod, valueExpressions,
            valueDerivExpressions, valueGradientExpressions, valueParamDerivExpressions,
            valueNames, valueTypes,energyExpressions, energyDerivExpressions, 
            energyGradientExpressions, energyParamDerivExpressions, energyTypes, particleParameterNames);
    bool periodic = (nonbondedMethod == CutoffPeriodic);
    if (periodic)
        ixn.setPeriodic(extractBoxVectors(context));
    if (nonbondedMethod != NoCutoff) {
        vector<set<int> > empty(context.getSystem().getNumParticles()); // Don't omit exclusions from the neighbor list
        computeNeighborListVoxelHash(*neighborList, numParticles, posData, empty, extractBoxVectors(context), periodic, nonbondedCutoff, 0.0);
        ixn.setUseCutoff(nonbondedCutoff, *neighborList);
    }
    map<string, double> globalParameters;
    for (auto& name : globalParameterNames)
        globalParameters[name] = context.getParameter(name);
    vector<double> energyParamDerivValues(energyParamDerivNames.size()+1, 0.0);

    integralMethod->BeforeComputation(context, posData);
    ixn.calculateIxn(posData, particleParamArray, exclusions, globalParameters, forceData, includeEnergy ? &energy : NULL, &energyParamDerivValues[0], context);
    integralMethod->FinishComputation(context, posData);

    map<string, double>& energyParamDerivs = extractEnergyParameterDerivatives(context);
    for (int i = 0; i < energyParamDerivNames.size(); i++)
        energyParamDerivs[energyParamDerivNames[i]] += energyParamDerivValues[i];
    return energy;
}

void ReferenceCalcCharmmGBMVForceKernel::copyParametersToContext(ContextImpl& context, const CharmmGBMVForce& force) {
    if (numParticles != force.getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the values.

    int numParameters = force.getNumPerParticleParameters();
    vector<double> params;
    for (int i = 0; i < numParticles; ++i) {
        vector<double> parameters;
        force.getParticleParameters(i, parameters);
        for (int j = 0; j < numParameters; j++)
            particleParamArray[i][j] = parameters[j];
    }
}

// ***************************************************************************
ReferenceCalcCharmmGBSWForceKernel::~ReferenceCalcCharmmGBSWForceKernel() {
    if(gbsw) delete gbsw;
}

void ReferenceCalcCharmmGBSWForceKernel::initialize(const System& system, const CharmmGBSWForce& force) {
    int numParticles = system.getNumParticles();
    charges.resize(numParticles);
    vector<double> atomicRadii(numParticles);
    vector<double> scaleFactors(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        double charge, radius, scalingFactor;
        force.getParticleParameters(i, charge, radius, scalingFactor);
        charges[i] = charge;
        atomicRadii[i] = radius;
        scaleFactors[i] = scalingFactor;
    }    
    gbsw = new CharmmReferenceGBSW(numParticles);
    gbsw->setAtomicRadii(atomicRadii);
    gbsw->setScaledRadiusFactors(scaleFactors);
    gbsw->setSolventDielectric(force.getSolventDielectric());
    gbsw->setSoluteDielectric(force.getSoluteDielectric());
    if (force.getNonbondedMethod() != CharmmGBSWForce::NoCutoff){
        OpenMM::NeighborList* neighborList = new NeighborList();
        gbsw->setUseCutoff(force.getCutoffDistance());
        gbsw->setNeighborList(*neighborList);
        if(force.getNonbondedMethod() == CharmmGBSWForce::CutoffPeriodic)
            gbsw->setPeriodic();
    }
    else {
        OpenMM::NeighborList* neighborList = NULL;
        gbsw->setNeighborList(*neighborList);
        gbsw->setNoCutoff();
        gbsw->setNoPeriodic();
    }
    return;
}

double ReferenceCalcCharmmGBSWForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    if (gbsw->getUseCutoff() && gbsw->getPeriodic())
        gbsw->setPeriodic(extractBoxVectors(context));
    if (gbsw->getUseCutoff()) {
        vector<set<int> > empty(context.getSystem().getNumParticles()); // Don't omit exclusions from the neighbor list
        OpenMM::NeighborList* neighborList = gbsw->getNeighborList();
        computeNeighborListVoxelHash(*neighborList, context.getSystem().getNumParticles(), posData, empty, extractBoxVectors(context), gbsw->getPeriodic(), gbsw->getCutoffDistance(), 0.0);
    }
    return gbsw->computeEnergyForces(posData, charges, forceData);
}

void ReferenceCalcCharmmGBSWForceKernel::copyParametersToContext(ContextImpl& context, const CharmmGBSWForce& force) {
    int numParticles = force.getNumParticles();
    if (numParticles != gbsw->getAtomicRadii().size())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the values.

    vector<double> atomicRadii(numParticles);
    vector<double> scaleFactors(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        double charge, radius, scalingFactor;
        force.getParticleParameters(i, charge, radius, scalingFactor);
        charges[i] = charge;
        atomicRadii[i] = radius;
        scaleFactors[i] = scalingFactor;
    }
    gbsw->setAtomicRadii(atomicRadii);
    gbsw->setScaledRadiusFactors(scaleFactors);
    return;
}

