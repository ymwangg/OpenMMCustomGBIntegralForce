/* -------------------------------------------------------------------------- *
 *                               OpenMMCharmm                                 *
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
#include "CustomGBIntegral.h"
#include "GBSWIntegral.h"

#include "CharmmCudaKernels.h"
#include "CudaKernels.h"
#include "CudaForceInfo.h"
#include "openmm/Context.h"
#include "lepton/CustomFunction.h"
#include "lepton/ExpressionTreeNode.h"
#include "lepton/Operation.h"
#include "lepton/Parser.h"
#include "CudaBondedUtilities.h"
#include "CudaExpressionUtilities.h"
#include "CudaIntegrationUtilities.h"
#include "CudaNonbondedUtilities.h"
#include "CudaCharmmKernelSources.h"
#include <cstdio>

#define REAL double
#define REAL3 double3
#define REAL4 double4

using namespace OpenMM;
using namespace std;
using namespace Lepton;

static bool isZeroExpression(const Lepton::ParsedExpression& expression) {
    const Lepton::Operation& op = expression.getRootNode().getOperation();
    if (op.getId() != Lepton::Operation::CONSTANT)
        return false;
    return (dynamic_cast<const Lepton::Operation::Constant&>(op).getValue() == 0.0);
}

static bool usesVariable(const Lepton::ExpressionTreeNode& node, const string& variable) {
    const Lepton::Operation& op = node.getOperation();
    if (op.getId() == Lepton::Operation::VARIABLE && op.getName() == variable)
        return true;
    for (auto& child : node.getChildren())
        if (usesVariable(child, variable))
            return true;
    return false;
}

static bool usesVariable(const Lepton::ParsedExpression& expression, const string& variable) {
    return usesVariable(expression.getRootNode(), variable);
}

static pair<ExpressionTreeNode, string> makeVariable(const string& name, const string& value) {
    return make_pair(ExpressionTreeNode(new Operation::Variable(name)), value);
}

static void replaceFunctionsInExpression(map<string, CustomFunction*>& functions, ExpressionProgram& expression) {
    for (int i = 0; i < expression.getNumOperations(); i++) {
        if (expression.getOperation(i).getId() == Operation::CUSTOM) {
            const Operation::Custom& op = dynamic_cast<const Operation::Custom&>(expression.getOperation(i));
            expression.setOperation(i, new Operation::Custom(op.getName(), functions[op.getName()]->clone(), op.getDerivOrder()));
        }
    }
}

class CudaCalcCharmmGBMVForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const CharmmGBMVForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        vector<double> params1;
        vector<double> params2;
        force.getParticleParameters(particle1, params1);
        force.getParticleParameters(particle2, params2);
        for (int i = 0; i < (int) params1.size(); i++)
            if (params1[i] != params2[i])
                return false;
        return true;
    }
    int getNumParticleGroups() {
        return force.getNumExclusions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        force.getExclusionParticles(index, particle1, particle2);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    const CharmmGBMVForce& force;
};

CudaCalcCharmmGBMVForceKernel::~CudaCalcCharmmGBMVForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
    if (computedValues != NULL)
        delete computedValues;
    if (energyDerivs != NULL)
        delete energyDerivs;
    if (energyDerivChain != NULL)
        delete energyDerivChain;
    for (auto d : dValuedParam)
        delete d;
}

void CudaCalcCharmmGBMVForceKernel::initialize(const System& system, const CharmmGBMVForce& force) {
    cu.setAsCurrent();

    numAtoms = force.getNumParticles();
    integralMethod = new GBSWIntegral();
    integralMethod->initialize(system,force);
    integralGradients.initialize<REAL3>(cu, force.getNumGBIntegrals()*force.getNumParticles()*force.getNumParticles(), "CharmmGBMVIntegralGradients");
    /*
    unsigned int num_points = 150*150*150;
    lookupTable.initialize<float*>(cu, num_points, "CharmmGBMVLookupTable");
    lookupTableNumAtoms.initialize<int>(cu, num_points, "CharmmGBMVLookupTableNumAtoms");
    CUmodule module = cu.createModule(CudaCharmmKernelSources::lookupTable);
    lookupTableKernel = cu.getKernel(module, "computeLookupTable");
    */
    if (cu.getPlatformData().contexts.size() > 1) 
        throw OpenMMException("CharmmGBMVForce does not support using multiple CUDA devices");
    cutoff = force.getCutoffDistance();
    bool useExclusionsForValue = false;
    numComputedValues = force.getNumComputedValues();
    numComputedIntegrals = force.getNumGBIntegrals();
    vector<string> computedValueNames(force.getNumComputedValues());
    vector<string> computedIntegralNames(force.getNumGBIntegrals());
    vector<string> computedValueExpressions(force.getNumComputedValues());
    if (force.getNumComputedValues() > 0) {
        CharmmGBMVForce::ComputationType type;
        //if (type == CharmmGBMVForce::SingleParticle)
            //throw OpenMMException("CudaPlatform requires that the first computed value for a CharmmGBMVForce be of type ParticlePair or ParticlePairNoExclusions.");
        useExclusionsForValue = (type == CharmmGBMVForce::ParticlePair);
        for (int i = 0; i < force.getNumComputedValues(); i++) {
            force.getComputedValueParameters(i, computedValueNames[i], computedValueExpressions[i], type);
            //if (type != CharmmGBMVForce::SingleParticle)
                //throw OpenMMException("CudaPlatform requires that a CharmmGBMVForce only have one computed value of type ParticlePair or ParticlePairNoExclusions.");
        }
        for (int i = 0; i < force.getNumGBIntegrals(); i++){
            force.getGBIntegralParameters(i, computedIntegralNames[i]);
        }
    }
    int forceIndex;
    for (forceIndex = 0; forceIndex < system.getNumForces() && &system.getForce(forceIndex) != &force; ++forceIndex)
        ;
    string prefix = "custom"+cu.intToString(forceIndex)+"_";

    // Record parameters and exclusions.
    int numParticles = force.getNumParticles();
    int paddedNumParticles = cu.getPaddedNumAtoms(); //
    int numParams = force.getNumPerParticleParameters(); //number of per particle parameters
    params = new CudaParameterSet(cu, force.getNumPerParticleParameters(), paddedNumParticles, "CharmmGBMVParameters", true);
    computedIntegrals = new CudaParameterSet(cu, force.getNumGBIntegrals(), paddedNumParticles, "CharmmGBMVComputedIntegrals", true, cu.getUseDoublePrecision());
    computedValues = new CudaParameterSet(cu, force.getNumComputedValues(), paddedNumParticles, "CharmmGBMVComputedValues", true, cu.getUseDoublePrecision());

    if (force.getNumGlobalParameters() > 0)
        globals.initialize<float>(cu, force.getNumGlobalParameters(), "CharmmGBMVGlobals");
    vector<vector<float> > paramVector(paddedNumParticles, vector<float>(numParams, 0));
    vector<vector<int> > exclusionList(numParticles);
    for (int i = 0; i < numParticles; i++) {
        vector<double> parameters;
        force.getParticleParameters(i, parameters);
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
        exclusionList[i].push_back(i);
    }
    for (int i = 0; i < force.getNumExclusions(); i++) {
        int particle1, particle2;
        force.getExclusionParticles(i, particle1, particle2);
        exclusionList[particle1].push_back(particle2);
        exclusionList[particle2].push_back(particle1);
    }
    params->setParameterValues(paramVector);

    // Record the tabulated functions.

    map<string, Lepton::CustomFunction*> functions;
    vector<pair<string, string> > functionDefinitions;
    vector<const TabulatedFunction*> functionList;
    stringstream tableArgs;
    tabulatedFunctions.resize(force.getNumTabulatedFunctions());
    for (int i = 0; i < force.getNumTabulatedFunctions(); i++) {
        functionList.push_back(&force.getTabulatedFunction(i));
        string name = force.getTabulatedFunctionName(i);
        string arrayName = prefix+"table"+cu.intToString(i);
        functionDefinitions.push_back(make_pair(name, arrayName));
        functions[name] = cu.getExpressionUtilities().getFunctionPlaceholder(force.getTabulatedFunction(i));
        int width;
        vector<float> f = cu.getExpressionUtilities().computeFunctionCoefficients(force.getTabulatedFunction(i), width);
        tabulatedFunctions[i].initialize<float>(cu, f.size(), "TabulatedFunction");
        tabulatedFunctions[i].upload(f);
        cu.getNonbondedUtilities().addArgument(CudaNonbondedUtilities::ParameterInfo(arrayName, "float", width, width*sizeof(float), tabulatedFunctions[i].getDevicePointer()));
        tableArgs << ", const float";
        if (width > 1)
            tableArgs << width;
        tableArgs << "* __restrict__ " << arrayName;
    }

    // Record the global parameters.

    globalParamNames.resize(force.getNumGlobalParameters());
    globalParamValues.resize(force.getNumGlobalParameters());
    for (int i = 0; i < force.getNumGlobalParameters(); i++) {
        globalParamNames[i] = force.getGlobalParameterName(i);
        globalParamValues[i] = (float) force.getGlobalParameterDefaultValue(i);
    }
    if (globals.isInitialized())
        globals.upload(globalParamValues);
    // Record derivatives of expressions needed for the chain rule terms.

    vector<vector<Lepton::ParsedExpression> > valueGradientExpressions(force.getNumComputedValues());
    vector<vector<Lepton::ParsedExpression> > valueDerivExpressions(force.getNumComputedValues()+force.getNumGBIntegrals());
    vector<vector<Lepton::ParsedExpression> > valueParamDerivExpressions(force.getNumComputedValues());
    needParameterGradient = false;
    for (int i = 0; i < force.getNumComputedValues(); i++) {
        Lepton::ParsedExpression ex = Lepton::Parser::parse(computedValueExpressions[i], functions).optimize();
        valueGradientExpressions[i].push_back(ex.differentiate("x").optimize());
        valueGradientExpressions[i].push_back(ex.differentiate("y").optimize());
        valueGradientExpressions[i].push_back(ex.differentiate("z").optimize());
        if (!isZeroExpression(valueGradientExpressions[i][0]) || !isZeroExpression(valueGradientExpressions[i][1]) || !isZeroExpression(valueGradientExpressions[i][2]))
            needParameterGradient = true;
        for (int j = 0; j < force.getNumGBIntegrals(); j++)
            valueDerivExpressions[i].push_back(ex.differentiate(computedIntegralNames[j]).optimize());
        for (int j = 0; j < i; j++)
            valueDerivExpressions[i].push_back(ex.differentiate(computedValueNames[j]).optimize());
        for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++)
            valueParamDerivExpressions[i].push_back(ex.differentiate(force.getEnergyParameterDerivativeName(j)).optimize());
    }

    // energy derivatives with respect to integrals, values, and global parameters
    vector<vector<Lepton::ParsedExpression> > energyDerivExpressions(force.getNumEnergyTerms());
    vector<vector<Lepton::ParsedExpression> > energyParamDerivExpressions(force.getNumEnergyTerms());
    vector<bool> needChainForIntegral(force.getNumGBIntegrals(), false);
    vector<bool> needChainForValue(force.getNumComputedValues(), false);
    for (int i = 0; i < force.getNumEnergyTerms(); i++) {
        string expression;
        CharmmGBMVForce::ComputationType type;
        force.getEnergyTermParameters(i, expression, type);
        Lepton::ParsedExpression ex = Lepton::Parser::parse(expression, functions).optimize();
        // differentiate with respect to integrals
        for (int j = 0; j < numComputedIntegrals; j++) {
            if (type == CharmmGBMVForce::SingleParticle) {
                energyDerivExpressions[i].push_back(ex.differentiate(computedIntegralNames[j]).optimize());
                if (!isZeroExpression(energyDerivExpressions[i].back()))
                    needChainForIntegral[j] = true;
            }
            else {
                energyDerivExpressions[i].push_back(ex.differentiate(computedIntegralNames[j]+"1").optimize());
                if (!isZeroExpression(energyDerivExpressions[i].back()))
                    needChainForIntegral[j] = true;
                energyDerivExpressions[i].push_back(ex.differentiate(computedIntegralNames[j]+"2").optimize());
                if (!isZeroExpression(energyDerivExpressions[i].back()))
                    needChainForIntegral[j] = true;
            }
        }

        // differentiate with respect to values
        for (int j = 0; j < numComputedValues; j++) {
            if (type == CharmmGBMVForce::SingleParticle) {
                energyDerivExpressions[i].push_back(ex.differentiate(computedValueNames[j]).optimize());
                if (!isZeroExpression(energyDerivExpressions[i].back()))
                    needChainForValue[j] = true;
            }
            else {
                energyDerivExpressions[i].push_back(ex.differentiate(computedValueNames[j]+"1").optimize());
                if (!isZeroExpression(energyDerivExpressions[i].back()))
                    needChainForValue[j] = true;
                energyDerivExpressions[i].push_back(ex.differentiate(computedValueNames[j]+"2").optimize());
                if (!isZeroExpression(energyDerivExpressions[i].back()))
                    needChainForValue[j] = true;
            }
        }
        for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++)
            energyParamDerivExpressions[i].push_back(ex.differentiate(force.getEnergyParameterDerivativeName(j)).optimize());
    }

    // ???
    longEnergyDerivs.initialize<long long>(cu, (force.getNumComputedValues()+force.getNumGBIntegrals())*cu.getPaddedNumAtoms(), "CharmmGBMVLongEnergyDerivatives");
    energyDerivs = new CudaParameterSet(cu, (force.getNumComputedValues()+force.getNumGBIntegrals()), cu.getPaddedNumAtoms(), "CharmmGBMVEnergyDerivatives", true);
    energyDerivChain = new CudaParameterSet(cu, (force.getNumComputedValues()+force.getNumGBIntegrals()), cu.getPaddedNumAtoms(), "CharmmGBMVEnergyDerivativeChain", true);
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    needEnergyParamDerivs = (force.getNumEnergyParameterDerivatives() > 0);
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        dValuedParam.push_back(new CudaParameterSet(cu, force.getNumComputedValues(), cu.getPaddedNumAtoms(), "dValuedParam", true, cu.getUseDoublePrecision()));
        string name = force.getEnergyParameterDerivativeName(i);
        cu.addEnergyParameterDerivative(name);
    }

    // Create the kernels.

    bool useCutoff = (force.getNonbondedMethod() != CharmmGBMVForce::NoCutoff);
    bool usePeriodic = (force.getNonbondedMethod() != CharmmGBMVForce::NoCutoff && force.getNonbondedMethod() != CharmmGBMVForce::CutoffNonPeriodic);
    {
        // calculate per particle integrals and gradients
    }
    {
        // Create the kernel to calculate per particle values and 
        // total derivatives dValuedParam

        stringstream reductionSource, extraArgs;
        // global parameters arg
        if (force.getNumGlobalParameters() > 0)
            extraArgs << ", const float* globals";

        // per particle parameters
        for (int i = 0; i < (int) params->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = params->getBuffers()[i];
            string paramName = "params"+cu.intToString(i+1);
            extraArgs << ", const " << buffer.getType() << "* __restrict__ " << paramName;
        }

        // integrals
        reductionSource << "\n";
        for (int i = 0; i < (int) computedIntegrals->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = computedIntegrals->getBuffers()[i];
            string integralName = "integrals"+cu.intToString(i+1);
            extraArgs << ", const " << buffer.getType() << "* __restrict__ global_" << integralName;
            reductionSource << buffer.getType() << " local_" << integralName << " = global_" << integralName << "[index];\n";
        }
        
        // values
        for (int i = 0; i < (int) computedValues->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = computedValues->getBuffers()[i];
            string valueName = "values"+cu.intToString(i+1);
            extraArgs << ", " << buffer.getType() << "* __restrict__ global_" << valueName;
            reductionSource << buffer.getType() << " local_" << valueName << ";\n";
        }

        // dValuedParam
        for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
            for (int j = 0; j < dValuedParam[i]->getBuffers().size(); j++)
                extraArgs << ", real* __restrict__ global_dValuedParam_" << j << "_" << i;
        }

        // reduction code
        map<string, string> variables;
        variables["x"] = "pos.x";
        variables["y"] = "pos.y";
        variables["z"] = "pos.z";
        for (int i = 0; i < force.getNumPerParticleParameters(); i++)
            variables[force.getPerParticleParameterName(i)] = "params"+params->getParameterSuffix(i, "[index]");
        for (int i = 0; i < force.getNumGlobalParameters(); i++)
            variables[force.getGlobalParameterName(i)] = "globals["+cu.intToString(i)+"]";
        for (int i = 0; i < force.getNumGBIntegrals(); i++) 
            variables[computedIntegralNames[i]] = "local_integrals"+computedIntegrals->getParameterSuffix(i);
        // compute values
        // exclude the last value
        for (int i = 0; i < force.getNumComputedValues(); i++) {
            if (i!= 0)
                variables[computedValueNames[i-1]] = "local_values"+computedValues->getParameterSuffix(i-1);
            map<string, Lepton::ParsedExpression> valueExpressions;
            valueExpressions["local_values"+computedValues->getParameterSuffix(i)+" = "] = Lepton::Parser::parse(computedValueExpressions[i], functions).optimize();
            reductionSource << cu.getExpressionUtilities().createExpressions(valueExpressions, variables, functionList, functionDefinitions, "value"+cu.intToString(i)+"_temp");
        }

        for (int i = 0; i < (int) computedValues->getBuffers().size(); i++) {
            string valueName = "values"+cu.intToString(i+1);
            reductionSource << "global_" << valueName << "[index] = local_" << valueName << ";\n";
        }
        // calculate partial derivative
        if (needEnergyParamDerivs){
            map<string, Lepton::ParsedExpression> derivExpressions;
            for (int i = 0; i < force.getNumComputedValues(); i++) {
                for (int j = 0; j < valueParamDerivExpressions[i].size(); j++)
                    derivExpressions["real dValuedParam_"+cu.intToString(i)+"_"+cu.intToString(j)+" = "] = valueParamDerivExpressions[i][j];
                for (int j = 0; j < i; j++)
                    derivExpressions["real dVdV_"+cu.intToString(i)+"_"+cu.intToString(j)+" = "] = valueDerivExpressions[i][j+numComputedIntegrals];
            }
            reductionSource << cu.getExpressionUtilities().createExpressions(derivExpressions, variables, functionList, functionDefinitions, "derivChain_temp");
            // calculate total derivative dValue_dIntegral
            for (int i = 0; i < force.getNumComputedValues(); i++) {
                for (int j = 0; j < i; j++){
                    for (int k = 0; k < valueParamDerivExpressions[i].size(); k++)
                        reductionSource << "dValuedParam_" << i << "_" << k << " += dVdV_" << i << "_" << j << "*dValuedParam_" << j <<"_" << k << ";\n";
                }
                // save total derivative dValuedParam
                if (needEnergyParamDerivs){
                    for (int j = 0; j < valueParamDerivExpressions[i].size(); j++)
                        reductionSource << "global_dValuedParam_" << i << "_" << j << "[index] = dValuedParam_" << i << "_" << j << ";\n";
                }
            }
        }
        map<string, string> replacements;
        replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
        replacements["COMPUTE_VALUES"] = reductionSource.str();
        map<string, string> defines;
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        cout<<cu.replaceStrings(CudaCharmmKernelSources::customGBValuePerParticle, replacements)<<endl;
        CUmodule module = cu.createModule(cu.replaceStrings(CudaCharmmKernelSources::customGBValuePerParticle, replacements), defines);
        perParticleValueKernel = cu.getKernel(module, "computePerParticleValues");
    }
    {
        // Create the N2 energy kernel.
        // also calculate partial derivatives with respect to integrals and values

        vector<pair<ExpressionTreeNode, string> > variables;
        ExpressionTreeNode rnode(new Operation::Variable("r"));
        variables.push_back(make_pair(rnode, "r"));
        variables.push_back(make_pair(ExpressionTreeNode(new Operation::Square(), rnode), "r2"));
        variables.push_back(make_pair(ExpressionTreeNode(new Operation::Reciprocal(), rnode), "invR"));
        // add per particle parameters
        for (int i = 0; i < force.getNumPerParticleParameters(); i++) {
            const string& name = force.getPerParticleParameterName(i);
            variables.push_back(makeVariable(name+"1", "params"+params->getParameterSuffix(i, "1")));
            variables.push_back(makeVariable(name+"2", "params"+params->getParameterSuffix(i, "2")));
        }
        // add integrals
        for (int i = 0; i < force.getNumGBIntegrals(); i++) {
            variables.push_back(makeVariable(computedIntegralNames[i]+"1", "integrals"+computedIntegrals->getParameterSuffix(i, "1")));
            variables.push_back(makeVariable(computedIntegralNames[i]+"2", "integrals"+computedIntegrals->getParameterSuffix(i, "2")));
        }
        // add values
        for (int i = 0; i < force.getNumComputedValues(); i++) {
            variables.push_back(makeVariable(computedValueNames[i]+"1", "values"+computedValues->getParameterSuffix(i, "1")));
            variables.push_back(makeVariable(computedValueNames[i]+"2", "values"+computedValues->getParameterSuffix(i, "2")));
        }
        // add global parameters
        for (int i = 0; i < force.getNumGlobalParameters(); i++)
            variables.push_back(makeVariable(force.getGlobalParameterName(i), "globals["+cu.intToString(i)+"]"));
        // energy source
        stringstream n2EnergySource;
        bool anyExclusions = (force.getNumExclusions() > 0);
        for (int i = 0; i < force.getNumEnergyTerms(); i++) {
            string expression;
            CharmmGBMVForce::ComputationType type;
            force.getEnergyTermParameters(i, expression, type);
            if (type == CharmmGBMVForce::SingleParticle)
                continue;
            bool exclude = (anyExclusions && type == CharmmGBMVForce::ParticlePair);
            map<string, Lepton::ParsedExpression> n2EnergyExpressions;
            n2EnergyExpressions["tempEnergy += "] = Lepton::Parser::parse(expression, functions).optimize();
            n2EnergyExpressions["dEdR += "] = Lepton::Parser::parse(expression, functions).differentiate("r").optimize();
            // derivatives with respect to integrals
            for (int j = 0; j < force.getNumGBIntegrals(); j++) {
                if (needChainForIntegral[j]) {
                    string index = cu.intToString(j+1);
                    n2EnergyExpressions["/*"+cu.intToString(i+1)+"*/ deriv"+index+"_1 += "] = energyDerivExpressions[i][2*j];
                    n2EnergyExpressions["/*"+cu.intToString(i+1)+"*/ deriv"+index+"_2 += "] = energyDerivExpressions[i][2*j+1];
                }
            }
            // derivatives with respect to values
            for (int j = 0; j < force.getNumComputedValues(); j++) {
                if (needChainForValue[j]) {
                    string index = cu.intToString(j+numComputedIntegrals+1);
                    n2EnergyExpressions["/*"+cu.intToString(i+1)+"*/ deriv"+index+"_1 += "] = energyDerivExpressions[i][2*(j+numComputedIntegrals)];
                    n2EnergyExpressions["/*"+cu.intToString(i+1)+"*/ deriv"+index+"_2 += "] = energyDerivExpressions[i][2*(j+numComputedIntegrals)+1];
                }
            }
            // derivatives with respect to global parameters
            for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++)
                n2EnergyExpressions["energyParamDeriv"+cu.intToString(j)+" += interactionScale*"] = energyParamDerivExpressions[i][j];
            if (exclude)
                n2EnergySource << "if (!isExcluded) {\n";
            n2EnergySource << cu.getExpressionUtilities().createExpressions(n2EnergyExpressions, variables, functionList, functionDefinitions, "temp");
            if (exclude)
                n2EnergySource << "}\n";
        }
        map<string, string> replacements;
        string n2EnergyStr = n2EnergySource.str();
        replacements["COMPUTE_INTERACTION"] = n2EnergyStr;
        // load atom parameters
        stringstream extraArgs, atomParams, loadLocal1, loadLocal2, clearLocal, load1, load2, declare1, recordDeriv, storeDerivs1, storeDerivs2, initParamDerivs, saveParamDerivs;
        if (force.getNumGlobalParameters() > 0)
            extraArgs << ", const float* globals";
        pairEnergyUsesParam.resize(params->getBuffers().size(), false);
        int atomParamSize = 7; // number of values in struct
        for (int i = 0; i < (int) params->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = params->getBuffers()[i];
            string paramName = "params"+cu.intToString(i+1);
            if (n2EnergyStr.find(paramName+"1") != n2EnergyStr.npos || n2EnergyStr.find(paramName+"2") != n2EnergyStr.npos) {
                extraArgs << ", const " << buffer.getType() << "* __restrict__ global_" << paramName;
                atomParams << buffer.getType() << " " << paramName << ";\n";
                loadLocal1 << "localData[localAtomIndex]." << paramName << " = " << paramName << "1;\n";
                loadLocal2 << "localData[localAtomIndex]." << paramName << " = global_" << paramName << "[j];\n";
                load1 << buffer.getType() << " " << paramName << "1 = global_" << paramName << "[atom1];\n";
                load2 << buffer.getType() << " " << paramName << "2 = localData[atom2]." << paramName << ";\n";
                pairEnergyUsesParam[i] = true;
                atomParamSize += buffer.getNumComponents();
            }
        }
        // load integrals
        pairEnergyUsesIntegral.resize(computedIntegrals->getBuffers().size(), false);
        for (int i = 0; i < (int) computedIntegrals->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = computedIntegrals->getBuffers()[i];
            string integralName = "integrals"+cu.intToString(i+1);
            if (n2EnergyStr.find(integralName+"1") != n2EnergyStr.npos || n2EnergyStr.find(integralName+"2") != n2EnergyStr.npos) {
                extraArgs << ", const " << buffer.getType() << "* __restrict__ global_" << integralName;
                atomParams << buffer.getType() << " " << integralName << ";\n";
                loadLocal1 << "localData[localAtomIndex]." << integralName << " = " << integralName << "1;\n";
                loadLocal2 << "localData[localAtomIndex]." << integralName << " = global_" << integralName << "[j];\n";
                load1 << buffer.getType() << " " << integralName << "1 = global_" << integralName << "[atom1];\n";
                load2 << buffer.getType() << " " << integralName << "2 = localData[atom2]." << integralName << ";\n";
                pairEnergyUsesIntegral[i] = true;
                atomParamSize += buffer.getNumComponents();
            }
        }
        // load values
        pairEnergyUsesValue.resize(computedValues->getBuffers().size(), false);
        for (int i = 0; i < (int) computedValues->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = computedValues->getBuffers()[i];
            string valueName = "values"+cu.intToString(i+1);
            if (n2EnergyStr.find(valueName+"1") != n2EnergyStr.npos || n2EnergyStr.find(valueName+"2") != n2EnergyStr.npos) {
                extraArgs << ", const " << buffer.getType() << "* __restrict__ global_" << valueName;
                atomParams << buffer.getType() << " " << valueName << ";\n";
                loadLocal1 << "localData[localAtomIndex]." << valueName << " = " << valueName << "1;\n";
                loadLocal2 << "localData[localAtomIndex]." << valueName << " = global_" << valueName << "[j];\n";
                load1 << buffer.getType() << " " << valueName << "1 = global_" << valueName << "[atom1];\n";
                load2 << buffer.getType() << " " << valueName << "2 = localData[atom2]." << valueName << ";\n";
                pairEnergyUsesValue[i] = true;
                atomParamSize += buffer.getNumComponents();
            }
        }
        // store derivatives
        extraArgs << ", unsigned long long* __restrict__ derivBuffers";
        // declare dEdIntegral
        for (int i = 0; i < force.getNumGBIntegrals(); i++) {
            if(!pairEnergyUsesIntegral[i]) continue;
            string index = cu.intToString(i+1);
            atomParams << "real deriv" << index << ";\n";
            clearLocal << "localData[localAtomIndex].deriv" << index << " = 0;\n";
            declare1 << "real deriv" << index << "_1 = 0;\n";
            load2 << "real deriv" << index << "_2 = 0;\n";
            recordDeriv << "localData[atom2].deriv" << index << " += deriv" << index << "_2;\n";
            storeDerivs1 << "STORE_DERIVATIVE_1(" << index << ")\n";
            storeDerivs2 << "STORE_DERIVATIVE_2(" << index << ")\n";
            atomParamSize++;
        }
        // declare dEdValue
        for (int i = 0; i < force.getNumComputedValues(); i++) {
            if(!pairEnergyUsesValue[i]) continue;
            string index = cu.intToString(i + numComputedIntegrals + 1);
            atomParams << "real deriv" << index << ";\n";
            clearLocal << "localData[localAtomIndex].deriv" << index << " = 0;\n";
            declare1 << "real deriv" << index << "_1 = 0;\n";
            load2 << "real deriv" << index << "_2 = 0;\n";
            recordDeriv << "localData[atom2].deriv" << index << " += deriv" << index << "_2;\n";
            storeDerivs1 << "STORE_DERIVATIVE_1(" << index << ")\n";
            storeDerivs2 << "STORE_DERIVATIVE_2(" << index << ")\n";
            atomParamSize++;
        }
        if (needEnergyParamDerivs) {
            extraArgs << ", mixed* __restrict__ energyParamDerivs";
            const vector<string>& allParamDerivNames = cu.getEnergyParamDerivNames();
            int numDerivs = allParamDerivNames.size();
            for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
                initParamDerivs << "mixed energyParamDeriv" << i << " = 0;\n";
                for (int index = 0; index < numDerivs; index++)
                    if (allParamDerivNames[index] == force.getEnergyParameterDerivativeName(i))
                        saveParamDerivs << "energyParamDerivs[(blockIdx.x*blockDim.x+threadIdx.x)*" << numDerivs << "+" << index << "] += energyParamDeriv" << i << ";\n";
            }
        }
        replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
        replacements["ATOM_PARAMETER_DATA"] = atomParams.str();
        replacements["LOAD_LOCAL_PARAMETERS_FROM_1"] = loadLocal1.str();
        replacements["LOAD_LOCAL_PARAMETERS_FROM_GLOBAL"] = loadLocal2.str();
        replacements["CLEAR_LOCAL_DERIVATIVES"] = clearLocal.str();
        replacements["LOAD_ATOM1_PARAMETERS"] = load1.str();
        replacements["LOAD_ATOM2_PARAMETERS"] = load2.str();
        replacements["DECLARE_ATOM1_DERIVATIVES"] = declare1.str();
        replacements["RECORD_DERIVATIVE_2"] = recordDeriv.str();
        replacements["STORE_DERIVATIVES_1"] = storeDerivs1.str();
        replacements["STORE_DERIVATIVES_2"] = storeDerivs2.str();
        replacements["INIT_PARAM_DERIVS"] = initParamDerivs.str();
        replacements["SAVE_PARAM_DERIVS"] = saveParamDerivs.str();
        if (useCutoff)
            pairEnergyDefines["USE_CUTOFF"] = "1";
        if (usePeriodic)
            pairEnergyDefines["USE_PERIODIC"] = "1";
        if (anyExclusions)
            pairEnergyDefines["USE_EXCLUSIONS"] = "1";
        if (atomParamSize%2 != 0 && !cu.getUseDoublePrecision())
            pairEnergyDefines["NEED_PADDING"] = "1";
        pairEnergyDefines["THREAD_BLOCK_SIZE"] = cu.intToString(cu.getNonbondedUtilities().getForceThreadBlockSize());
        pairEnergyDefines["WARPS_PER_GROUP"] = cu.intToString(cu.getNonbondedUtilities().getForceThreadBlockSize()/CudaContext::TileSize);
        pairEnergyDefines["CUTOFF_SQUARED"] = cu.doubleToString(cutoff*cutoff);
        pairEnergyDefines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        pairEnergyDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pairEnergyDefines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
        pairEnergyDefines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
        pairEnergySrc = cu.replaceStrings(CudaCharmmKernelSources::customGBEnergyN2, replacements);
        cout<<pairEnergySrc<<endl;
    }
    {
        // Create the kernel to reduce the derivatives and calculate per-particle energy terms.

        stringstream compute, extraArgs, load, initParamDerivs, saveParamDerivs;
        if (force.getNumGlobalParameters() > 0)
            extraArgs << ", const float* globals";
        // per particle parameters
        for (int i = 0; i < (int) params->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = params->getBuffers()[i];
            string paramName = "params"+cu.intToString(i+1);
            extraArgs << ", const " << buffer.getType() << "* __restrict__ " << paramName;
        }
        // integrals
        for (int i = 0; i < (int) computedIntegrals->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = computedIntegrals->getBuffers()[i];
            string integralName = "integrals"+cu.intToString(i+1);
            extraArgs << ", const " << buffer.getType() << "* __restrict__ " << integralName;
        }
        extraArgs << ", const real3 * __restrict__ integralGradients";
        // values
        for (int i = 0; i < (int) computedValues->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = computedValues->getBuffers()[i];
            string valueName = "values"+cu.intToString(i+1);
            extraArgs << ", const " << buffer.getType() << "* __restrict__ " << valueName;
        }
        // derivatives
        for (int i = 0; i < (int) energyDerivs->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = energyDerivs->getBuffers()[i];
            string index = cu.intToString(i+1);
            extraArgs << ", " << buffer.getType() << "* __restrict__ derivBuffers" << index;
            compute << buffer.getType() << " deriv" << index << " = derivBuffers" << index << "[index];\n";
        }
        // convert long long to float for derivatives
        extraArgs << ", const long long* __restrict__ derivBuffersIn";
        for (int i = 0; i < energyDerivs->getNumParameters(); ++i)
            load << "derivBuffers" << energyDerivs->getParameterSuffix(i, "[index]") <<
                    " = RECIP(0x100000000)*derivBuffersIn[index+PADDED_NUM_ATOMS*" << cu.intToString(i) << "];\n";
        if (needEnergyParamDerivs) {
            extraArgs << ", mixed* __restrict__ energyParamDerivs";
            const vector<string>& allParamDerivNames = cu.getEnergyParamDerivNames();
            int numDerivs = allParamDerivNames.size();
            for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
                initParamDerivs << "mixed energyParamDeriv" << i << " = 0;\n";
                for (int index = 0; index < numDerivs; index++)
                    if (allParamDerivNames[index] == force.getEnergyParameterDerivativeName(i))
                        saveParamDerivs << "energyParamDerivs[(blockIdx.x*blockDim.x+threadIdx.x)*" << numDerivs << "+" << index << "] += energyParamDeriv" << i << ";\n";
            }
        }
        // Compute the various expressions.

        map<string, string> variables;
        variables["x"] = "pos.x";
        variables["y"] = "pos.y";
        variables["z"] = "pos.z";
        for (int i = 0; i < force.getNumPerParticleParameters(); i++)
            variables[force.getPerParticleParameterName(i)] = "params"+params->getParameterSuffix(i, "[index]");
        for (int i = 0; i < force.getNumGlobalParameters(); i++)
            variables[force.getGlobalParameterName(i)] = "globals["+cu.intToString(i)+"]";
        for (int i = 0; i < force.getNumGBIntegrals(); i++)
            variables[computedIntegralNames[i]] = "integrals"+computedIntegrals->getParameterSuffix(i, "[index]");
        for (int i = 0; i < force.getNumComputedValues(); i++)
            variables[computedValueNames[i]] = "values"+computedValues->getParameterSuffix(i, "[index]");
        map<string, Lepton::ParsedExpression> expressions;
        for (int i = 0; i < force.getNumEnergyTerms(); i++) {
            string expression;
            CharmmGBMVForce::ComputationType type;
            force.getEnergyTermParameters(i, expression, type);
            if (type != CharmmGBMVForce::SingleParticle)
                continue;
            Lepton::ParsedExpression parsed = Lepton::Parser::parse(expression, functions).optimize();
            expressions["/*"+cu.intToString(i+1)+"*/ energy += "] = parsed;
            for (int j = 0; j < force.getNumGBIntegrals(); j++)
                expressions["/*"+cu.intToString(i+1)+"*/ deriv"+energyDerivs->getParameterSuffix(j)+" += "] = energyDerivExpressions[i][j];
            for (int j = 0; j < force.getNumComputedValues(); j++)
                expressions["/*"+cu.intToString(i+1)+"*/ deriv"+energyDerivs->getParameterSuffix(j+numComputedIntegrals)+" += "] = energyDerivExpressions[i][j+numComputedIntegrals];
            Lepton::ParsedExpression gradx = parsed.differentiate("x").optimize();
            Lepton::ParsedExpression grady = parsed.differentiate("y").optimize();
            Lepton::ParsedExpression gradz = parsed.differentiate("z").optimize();
            if (!isZeroExpression(gradx))
                expressions["/*"+cu.intToString(i+1)+"*/ force.x -= "] = gradx;
            if (!isZeroExpression(grady))
                expressions["/*"+cu.intToString(i+1)+"*/ force.y -= "] = grady;
            if (!isZeroExpression(gradz))
                expressions["/*"+cu.intToString(i+1)+"*/ force.z -= "] = gradz;
            for (int j = 0; j < force.getNumEnergyParameterDerivatives(); j++)
                expressions["/*"+cu.intToString(i+1)+"*/ energyParamDeriv"+cu.intToString(j)+" += "] = energyParamDerivExpressions[i][j];
        }
        for (int i = 0; i < force.getNumComputedValues(); i++){
            for (int j = 0; j < force.getNumGBIntegrals(); j++){
                expressions["real dV"+cu.intToString(i)+"dI"+cu.intToString(j)+" = "] = valueDerivExpressions[i][j];
            }
            for (int j = 0; j < i; j++){
                expressions["real dV"+cu.intToString(i)+"dV"+cu.intToString(j)+" = "] = valueDerivExpressions[i][j+numComputedIntegrals];
            }
        }
        compute << cu.getExpressionUtilities().createExpressions(expressions, variables, functionList, functionDefinitions, "temp");

        // appy chain rule to dEdIntegral
        for (int i = 0; i < force.getNumComputedValues(); i++) {
            for (int k = 0; k < force.getNumGBIntegrals(); k++){
                string indexIntegral = cu.intToString(k+1);
                string indexValue = cu.intToString(i+numComputedIntegrals+1);
                for (int j = 0; j < i; j++){
                    compute << "dV" << cu.intToString(i) << "dI" << cu.intToString(k) << " += dV" << cu.intToString(i) << "dV" << cu.intToString(j) << " * dV" << cu.intToString(j) << "dI" << cu.intToString(k) << ";\n";
                }
                compute << "deriv" << indexIntegral << " += deriv"<< indexValue << " * dV" << cu.intToString(i) << "dI" << cu.intToString(k) << ";\n";
            }
        }
        for (int i = 0; i < force.getNumGBIntegrals(); i++){
            string index = cu.intToString(i+1);
            compute << "for (unsigned int i = 0; i < NUM_ATOMS; i++) {\n";
            compute << "int gradient_index = " << i << "*NUM_ATOMS*NUM_ATOMS + index*NUM_ATOMS + i;\n";
            compute << "real3 tmp_gradient = integralGradients[gradient_index];\n";
            //compute << "printf(\"%d %d %f %f %f\\n\",index,i,tmp_gradient.x,tmp_gradient.y,tmp_gradient.z);\n";
            compute << "real3 tmp_force;\n";
            compute << "tmp_force.x = -1.0*deriv"<<index<< "* tmp_gradient.x;\n";
            compute << "tmp_force.y = -1.0*deriv"<<index<< "* tmp_gradient.y;\n";
            compute << "tmp_force.z = -1.0*deriv"<<index<< "* tmp_gradient.z;\n";
            compute << "atomicAdd(&forceBuffers[i], static_cast<unsigned long long>((long long) (tmp_force.x*0x100000000)));\n";
            compute << "atomicAdd(&forceBuffers[i+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (tmp_force.y*0x100000000)));\n";
            compute << "atomicAdd(&forceBuffers[i+PADDED_NUM_ATOMS*2], static_cast<unsigned long long>((long long) (tmp_force.z*0x100000000)));\n";
            compute << "}\n";
        }

        // Record values.

        for (int i = 0; i < (int) energyDerivs->getBuffers().size(); i++) {
            string index = cu.intToString(i+1);
            compute << "derivBuffers" << index << "[index] = deriv" << index << ";\n";
        }
        compute << "forceBuffers[index] += (long long) (force.x*0x100000000);\n";
        compute << "forceBuffers[index+PADDED_NUM_ATOMS] += (long long) (force.y*0x100000000);\n";
        compute << "forceBuffers[index+PADDED_NUM_ATOMS*2] += (long long) (force.z*0x100000000);\n";

        map<string, string> replacements;
        replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
        replacements["LOAD_DERIVATIVES"] = load.str();
        replacements["COMPUTE_ENERGY"] = compute.str();
        replacements["INIT_PARAM_DERIVS"] = initParamDerivs.str();
        replacements["SAVE_PARAM_DERIVS"] = saveParamDerivs.str();
        map<string, string> defines;
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        cout<<cu.replaceStrings(CudaCharmmKernelSources::customGBEnergyPerParticle, replacements)<<endl;
        CUmodule module = cu.createModule(cu.replaceStrings(CudaCharmmKernelSources::customGBEnergyPerParticle, replacements), defines);
        perParticleEnergyKernel = cu.getKernel(module, "computePerParticleEnergy");
    }

    {
        // Create the kernel to compute chain rule terms for computed values that depend explicitly on particle coordinates, and for
        // derivatives with respect to global parameters.
        stringstream compute, extraArgs, initParamDerivs, saveParamDerivs;
        if (force.getNumGlobalParameters() > 0)
            extraArgs << ", const float* globals";
        for (int i = 0; i < (int) params->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = params->getBuffers()[i];
            string paramName = "params"+cu.intToString(i+1);
            extraArgs << ", const " << buffer.getType() << "* __restrict__ " << paramName;
        }
        for (int i = 0; i < (int) computedIntegrals->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = computedIntegrals->getBuffers()[i];
            string integralName = "integrals"+cu.intToString(i+1);
            extraArgs << ", const " << buffer.getType() << "* __restrict__ " << integralName;
        }
        for (int i = 0; i < (int) computedValues->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = computedValues->getBuffers()[i];
            string valueName = "values"+cu.intToString(i+numComputedIntegrals+1);
            extraArgs << ", const " << buffer.getType() << "* __restrict__ " << valueName;
        }
        for (int i = 0; i < (int) energyDerivs->getBuffers().size(); i++) {
            CudaNonbondedUtilities::ParameterInfo& buffer = energyDerivs->getBuffers()[i];
            string index = cu.intToString(i+1);
            extraArgs << ", " << buffer.getType() << "* __restrict__ derivBuffers" << index;
            compute << buffer.getType() << " deriv" << index << " = derivBuffers" << index << "[index];\n";
        }
        if (needEnergyParamDerivs) {
            extraArgs << ", mixed* __restrict__ energyParamDerivs";
            const vector<string>& allParamDerivNames = cu.getEnergyParamDerivNames();
            int numDerivs = allParamDerivNames.size();
            for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
                for (int j = 0; j < dValuedParam[i]->getBuffers().size(); j++)
                    extraArgs << ", real* __restrict__ dValuedParam_" << j << "_" << i;
                initParamDerivs << "mixed energyParamDeriv" << i << " = 0;\n";
                for (int index = 0; index < numDerivs; index++)
                    if (allParamDerivNames[index] == force.getEnergyParameterDerivativeName(i))
                        saveParamDerivs << "energyParamDerivs[(blockIdx.x*blockDim.x+threadIdx.x)*" << numDerivs << "+" << index << "] += energyParamDeriv" << i << ";\n";
            }
        }
        map<string, string> variables;
        variables["x"] = "pos.x";
        variables["y"] = "pos.y";
        variables["z"] = "pos.z";
        for (int i = 0; i < force.getNumPerParticleParameters(); i++)
            variables[force.getPerParticleParameterName(i)] = "params"+params->getParameterSuffix(i, "[index]");
        for (int i = 0; i < force.getNumGlobalParameters(); i++)
            variables[force.getGlobalParameterName(i)] = "globals["+cu.intToString(i)+"]";
        for (int i = 0; i < force.getNumGBIntegrals(); i++)
            variables[computedIntegralNames[i]] = "integrals"+computedIntegrals->getParameterSuffix(i, "[index]");
        for (int i = 0; i < force.getNumComputedValues(); i++)
            variables[computedValueNames[i]] = "values"+computedValues->getParameterSuffix(i, "[index]");
        if (needParameterGradient) {
            for (int i = 0; i < force.getNumComputedValues(); i++) {
                string is = cu.intToString(i);
                compute << "real3 dV"<<is<<"dR = make_real3(0);\n";
                for (int j = 0; j < i; j++) {
                    if (!isZeroExpression(valueDerivExpressions[i][j])) {
                        map<string, Lepton::ParsedExpression> derivExpressions;
                        string js = cu.intToString(j);
                        derivExpressions["real dV"+is+"dV"+js+" = "] = valueDerivExpressions[i][j];
                        compute << cu.getExpressionUtilities().createExpressions(derivExpressions, variables, functionList, functionDefinitions, "temp_"+is+"_"+js);
                        compute << "dV"<<is<<"dR += dV"<<is<<"dV"<<js<<"*dV"<<js<<"dR;\n";
                    }
                }
                map<string, Lepton::ParsedExpression> gradientExpressions;
                if (!isZeroExpression(valueGradientExpressions[i][0]))
                    gradientExpressions["dV"+is+"dR.x += "] = valueGradientExpressions[i][0];
                if (!isZeroExpression(valueGradientExpressions[i][1]))
                    gradientExpressions["dV"+is+"dR.y += "] = valueGradientExpressions[i][1];
                if (!isZeroExpression(valueGradientExpressions[i][2]))
                    gradientExpressions["dV"+is+"dR.z += "] = valueGradientExpressions[i][2];
                compute << cu.getExpressionUtilities().createExpressions(gradientExpressions, variables, functionList, functionDefinitions, "temp");
            }
            for (int i = 0; i < force.getNumComputedValues(); i++) {
                string is = cu.intToString(i);
                compute << "force -= deriv"<<energyDerivs->getParameterSuffix(i)<<"*dV"<<is<<"dR;\n";
            }
        }
        if (needEnergyParamDerivs)
            for (int i = 0; i < force.getNumComputedValues(); i++)
                for (int j = 0; j < dValuedParam.size(); j++)
                    compute << "energyParamDeriv"<<j<<" += deriv"<<energyDerivs->getParameterSuffix(i)<<"*dValuedParam_"<<i<<"_"<<j<<"[index];\n";
        map<string, string> replacements;
        replacements["PARAMETER_ARGUMENTS"] = extraArgs.str()+tableArgs.str();
        replacements["COMPUTE_FORCES"] = compute.str();
        replacements["INIT_PARAM_DERIVS"] = initParamDerivs.str();
        replacements["SAVE_PARAM_DERIVS"] = saveParamDerivs.str();
        map<string, string> defines;
        defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
        defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        CUmodule module = cu.createModule(CudaCharmmKernelSources::vectorOps+cu.replaceStrings(CudaCharmmKernelSources::customGBGradientChainRule, replacements), defines);
        gradientChainRuleKernel = cu.getKernel(module, "computeGradientChainRuleTerms");
        //cout<<CudaCharmmKernelSources::vectorOps+cu.replaceStrings(CudaCharmmKernelSources::customGBGradientChainRule, replacements)<<endl;
    }

    {
        // Create the code to calculate chain rule terms as part of the default nonbonded kernel.
        // deleted
        string source = "\n";
        cu.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, force.getNumExclusions() > 0, cutoff, exclusionList, source, force.getForceGroup());
    }
    info = new ForceInfo(force);
    cu.addForce(info);
    cu.addAutoclearBuffer(longEnergyDerivs);
    return;
}

double CudaCalcCharmmGBMVForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<REAL4> oldPosq(cu.getPaddedNumAtoms());
    cu.getPosq().download(oldPosq);
    std::vector<OpenMM::Vec3> posData(numAtoms,OpenMM::Vec3());
    for (int i=0; i<numAtoms; i++){
        posData[i][0] = oldPosq[i].x;
        posData[i][1] = oldPosq[i].y;
        posData[i][2] = oldPosq[i].z;
    }
    //computedIntegrals;integralGradients;
    integralMethod->BeforeComputation(context, posData);
    vector<vector<REAL> > tmpIntegrals;
    tmpIntegrals.resize(cu.getPaddedNumAtoms(),vector<REAL>(numComputedIntegrals));
    vector<REAL3> tmpIntegralGradients;
    tmpIntegralGradients.resize(numComputedIntegrals*numAtoms*numAtoms);

    for(int atomI = 0; atomI < numAtoms; ++atomI){
        vector<double> tmpValues;
        std::vector<std::vector<OpenMM::Vec3> > tmpGradients;
        integralMethod->evaluate(atomI, context, posData, tmpValues, tmpGradients, true);
        for(int i=0; i<numComputedIntegrals; ++i){
            tmpIntegrals[atomI][i]=tmpValues[i];
            for(int m = 0; m < numAtoms; ++m){
                tmpIntegralGradients[i*numAtoms*numAtoms + atomI*numAtoms + m].x = tmpGradients[i][m][0];
                tmpIntegralGradients[i*numAtoms*numAtoms + atomI*numAtoms + m].y = tmpGradients[i][m][1];
                tmpIntegralGradients[i*numAtoms*numAtoms + atomI*numAtoms + m].z = tmpGradients[i][m][2];
            }
        }
    } 

    /*
    for(int i = 0; i < numComputedIntegrals; ++i){
        for(int atomI=0; atomI<numAtoms; ++atomI){
            for(int atomJ=0; atomJ<numAtoms; ++atomJ){
                int idx = i*numAtoms*numAtoms + atomJ*numAtoms + atomI;
                float3 tmp = tmpIntegralGradients[idx];
                printf("%d %d %f %f %f\n",atomJ,atomI,tmp.x,tmp.y,tmp.z);
            }
        }
    }
    */
    computedIntegrals->setParameterValues(tmpIntegrals);
    integralGradients.upload(tmpIntegralGradients);
    
    /*
    lookupTableArgs.push_back(&cu.getPosq().getDevicePointer());
    lookupTable.getDevicePointer();
    lookupTableNumAtoms.getDevicePointer();
    lookupTableArgs.push_back(cu.getPeriodicBoxSizePointer());
    lookupTableArgs.push_back(cu.getInvPeriodicBoxSizePointer());
    lookupTableArgs.push_back(cu.getPeriodicBoxVecXPointer());
    lookupTableArgs.push_back(cu.getPeriodicBoxVecYPointer());
    lookupTableArgs.push_back(cu.getPeriodicBoxVecZPointer());
    cu.executeKernel(lookupTableKernel, &lookupTableArgs[0],cu.getPaddedNumAtoms());
    return 0.0;
    */
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;

        // These two kernels can't be compiled in initialize(), because the nonbonded utilities object
        // has not yet been initialized then.

        {
            //deleted
        }
        {
            int numExclusionTiles = cu.getNonbondedUtilities().getExclusionTiles().getSize();
            pairEnergyDefines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
            int numContexts = cu.getPlatformData().contexts.size();
            int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
            int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
            pairEnergyDefines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
            pairEnergyDefines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
            pairEnergyDefines["CUTOFF"] = cu.doubleToString(cutoff);
            CUmodule module = cu.createModule(CudaCharmmKernelSources::vectorOps+pairEnergySrc, pairEnergyDefines);
            pairEnergyKernel = cu.getKernel(module, "computeN2Energy");
            pairEnergySrc = "";
            pairEnergyDefines.clear();
        }
        // Set arguments for kernels.

        maxTiles = (nb.getUseCutoff() ? nb.getInteractingTiles().getSize() : cu.getNumAtomBlocks()*(cu.getNumAtomBlocks()+1)/2);
        perParticleValueArgs.push_back(&cu.getPosq().getDevicePointer());
        if (globals.isInitialized())
            perParticleValueArgs.push_back(&globals.getDevicePointer());
        for (auto& buffer : params->getBuffers())
            perParticleValueArgs.push_back(&buffer.getMemory());
        for (auto& buffer : computedIntegrals->getBuffers())
            perParticleValueArgs.push_back(&buffer.getMemory());
        for (auto& buffer : computedValues->getBuffers())
            perParticleValueArgs.push_back(&buffer.getMemory());
        for (int i = 0; i < dValuedParam.size(); i++) {
            //perParticleValueArgs.push_back(&dValue0dParam[i].getDevicePointer());
            for (int j = 0; j < dValuedParam[i]->getBuffers().size(); j++)
                perParticleValueArgs.push_back(&dValuedParam[i]->getBuffers()[j].getMemory());
        }
        for (auto& function : tabulatedFunctions)
            perParticleValueArgs.push_back(&function.getDevicePointer());
        pairEnergyArgs.push_back(&cu.getForce().getDevicePointer());
        pairEnergyArgs.push_back(&cu.getEnergyBuffer().getDevicePointer());
        pairEnergyArgs.push_back(&cu.getPosq().getDevicePointer());
        pairEnergyArgs.push_back(&cu.getNonbondedUtilities().getExclusions().getDevicePointer());
        pairEnergyArgs.push_back(&cu.getNonbondedUtilities().getExclusionTiles().getDevicePointer());
        pairEnergyArgs.push_back(NULL);
        if (nb.getUseCutoff()) {
            pairEnergyArgs.push_back(&nb.getInteractingTiles().getDevicePointer());
            pairEnergyArgs.push_back(&nb.getInteractionCount().getDevicePointer());
            pairEnergyArgs.push_back(cu.getPeriodicBoxSizePointer());
            pairEnergyArgs.push_back(cu.getInvPeriodicBoxSizePointer());
            pairEnergyArgs.push_back(cu.getPeriodicBoxVecXPointer());
            pairEnergyArgs.push_back(cu.getPeriodicBoxVecYPointer());
            pairEnergyArgs.push_back(cu.getPeriodicBoxVecZPointer());
            pairEnergyArgs.push_back(&maxTiles);
            pairEnergyArgs.push_back(&nb.getBlockCenters().getDevicePointer());
            pairEnergyArgs.push_back(&nb.getBlockBoundingBoxes().getDevicePointer());
            pairEnergyArgs.push_back(&nb.getInteractingAtoms().getDevicePointer());
        }
        else
            pairEnergyArgs.push_back(&maxTiles);
        if (globals.isInitialized())
            pairEnergyArgs.push_back(&globals.getDevicePointer());
        for (int i = 0; i < (int) params->getBuffers().size(); i++) {
            if (pairEnergyUsesParam[i])
                pairEnergyArgs.push_back(&params->getBuffers()[i].getMemory());
        }
        for (int i = 0; i < (int) computedIntegrals->getBuffers().size(); i++) {
            if (pairEnergyUsesIntegral[i])
                pairEnergyArgs.push_back(&computedIntegrals->getBuffers()[i].getMemory());
        }
        for (int i = 0; i < (int) computedValues->getBuffers().size(); i++) {
            if (pairEnergyUsesValue[i])
                pairEnergyArgs.push_back(&computedValues->getBuffers()[i].getMemory());
        }
        pairEnergyArgs.push_back(&longEnergyDerivs.getDevicePointer());
        if (needEnergyParamDerivs)
            pairEnergyArgs.push_back(&cu.getEnergyParamDerivBuffer().getDevicePointer());
        for (auto& function : tabulatedFunctions)
            pairEnergyArgs.push_back(&function.getDevicePointer());
        perParticleEnergyArgs.push_back(&cu.getForce().getDevicePointer());
        perParticleEnergyArgs.push_back(&cu.getEnergyBuffer().getDevicePointer());
        perParticleEnergyArgs.push_back(&cu.getPosq().getDevicePointer());
        if (globals.isInitialized())
            perParticleEnergyArgs.push_back(&globals.getDevicePointer());
        for (auto& buffer : params->getBuffers())
            perParticleEnergyArgs.push_back(&buffer.getMemory());
        for (auto& buffer : computedIntegrals->getBuffers())
            perParticleEnergyArgs.push_back(&buffer.getMemory());
        perParticleEnergyArgs.push_back(&integralGradients.getDevicePointer());
        for (auto& buffer : computedValues->getBuffers())
            perParticleEnergyArgs.push_back(&buffer.getMemory());
        for (auto& buffer : energyDerivs->getBuffers())
            perParticleEnergyArgs.push_back(&buffer.getMemory());
        perParticleEnergyArgs.push_back(&longEnergyDerivs.getDevicePointer());
        if (needEnergyParamDerivs)
            perParticleEnergyArgs.push_back(&cu.getEnergyParamDerivBuffer().getDevicePointer());
        for (auto& function : tabulatedFunctions)
            perParticleEnergyArgs.push_back(&function.getDevicePointer());
        if (needParameterGradient || needEnergyParamDerivs) {
            gradientChainRuleArgs.push_back(&cu.getForce().getDevicePointer());
            gradientChainRuleArgs.push_back(&cu.getPosq().getDevicePointer());
            if (globals.isInitialized())
                gradientChainRuleArgs.push_back(&globals.getDevicePointer());
            for (auto& buffer : params->getBuffers())
                gradientChainRuleArgs.push_back(&buffer.getMemory());
            for (auto& buffer : computedValues->getBuffers())
                gradientChainRuleArgs.push_back(&buffer.getMemory());
            for (auto& buffer : energyDerivs->getBuffers())
                gradientChainRuleArgs.push_back(&buffer.getMemory());
            if (needEnergyParamDerivs) {
                gradientChainRuleArgs.push_back(&cu.getEnergyParamDerivBuffer().getDevicePointer());
                for (auto d : dValuedParam)
                    for (auto& buffer : d->getBuffers())
                        gradientChainRuleArgs.push_back(&buffer.getMemory());
            }
        }
    }
    if (globals.isInitialized()) {
        bool changed = false;
        for (int i = 0; i < (int) globalParamNames.size(); i++) {
            float value = (float) context.getParameter(globalParamNames[i]);
            if (value != globalParamValues[i])
                changed = true;
            globalParamValues[i] = value;
        }
        if (changed)
            globals.upload(globalParamValues);
    }
    pairEnergyArgs[5] = &includeEnergy;
    if (nb.getUseCutoff()) {
        if (maxTiles < nb.getInteractingTiles().getSize()) {
            maxTiles = nb.getInteractingTiles().getSize();
            pairEnergyArgs[6] = &nb.getInteractingTiles().getDevicePointer();
            pairEnergyArgs[16] = &nb.getInteractingAtoms().getDevicePointer();
        }
    }
    vector<vector<REAL> > values;
    cu.executeKernel(perParticleValueKernel, &perParticleValueArgs[0], cu.getPaddedNumAtoms());
    cu.executeKernel(pairEnergyKernel, &pairEnergyArgs[0], nb.getNumForceThreadBlocks()*nb.getForceThreadBlockSize(), nb.getForceThreadBlockSize());
    cu.executeKernel(perParticleEnergyKernel, &perParticleEnergyArgs[0], cu.getPaddedNumAtoms());
    if (needParameterGradient || needEnergyParamDerivs)
        cu.executeKernel(gradientChainRuleKernel, &gradientChainRuleArgs[0], cu.getPaddedNumAtoms());
    /*
    computedIntegrals->getParameterValues(values);
    for(int i=0; i<values.size(); ++i){
        cout<<posData[i]<<" ";
        for(auto &c : values[i]) cout<<i<<" "<<c<<" ";
        cout<<endl;
    }
    */
    return 0.0;
}

void CudaCalcCharmmGBMVForceKernel::copyParametersToContext(ContextImpl& context, const CharmmGBMVForce& force) {
    cu.setAsCurrent();
    int numParticles = force.getNumParticles();
    if (numParticles != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the per-particle parameters.

    vector<vector<float> > paramVector(cu.getPaddedNumAtoms(), vector<float>(force.getNumPerParticleParameters(), 0));
    vector<double> parameters;
    for (int i = 0; i < numParticles; i++) {
        force.getParticleParameters(i, parameters);
        for (int j = 0; j < (int) parameters.size(); j++)
            paramVector[i][j] = (float) parameters[j];
    }
    params->setParameterValues(paramVector);

    // Mark that the current reordering may be invalid.

    cu.invalidateMolecules();
    return;
}

