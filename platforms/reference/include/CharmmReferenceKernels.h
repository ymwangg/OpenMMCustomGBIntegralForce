
/* Portions copyright (c) 2006-2016 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __CharmmReferenceKernels_H__
#define __CharmmReferenceKernels_H__

#include "GBSWIntegral.h"
#include "openmm/CharmmGBMVForce.h"
#include "openmm/CharmmGBSWForce.h"
#include "openmm/charmmKernels.h"
#include <vector>

#include "ReferencePlatform.h"
#include "SimTKOpenMMRealType.h"
#include "ReferenceNeighborList.h"
#include "lepton/CompiledExpression.h"
#include "lepton/CustomFunction.h"


namespace OpenMM {

class CharmmReferenceGBMV;
class CharmmReferenceGBSW;
/**
 * This kernel is invoked by CharmmGBMVForce to calculate the forces acting on the system.
 */
class ReferenceCalcCharmmGBMVForceKernel : public CalcCharmmGBMVForceKernel {
public:
    ReferenceCalcCharmmGBMVForceKernel(std::string name, const Platform& platform) : CalcCharmmGBMVForceKernel(name, platform) {
    }
    ~ReferenceCalcCharmmGBMVForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the CharmmGBMVForce this kernel will be used for
     */
    void initialize(const System& system, const CharmmGBMVForce& force);
    /**  
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**  
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CharmmGBMVForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CharmmGBMVForce& force);
    double validateIntegral(ContextImpl& context);
private:
    int numParticles;
    bool isPeriodic;
    double **particleParamArray;
    double nonbondedCutoff;
    std::vector<std::set<int> > exclusions;
    std::vector<std::string> particleParameterNames, globalParameterNames, energyParamDerivNames, valueNames, integralNames;
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
    NonbondedMethod nonbondedMethod;
    NeighborList* neighborList;
    CustomGBIntegral* integralMethod;
};

/**
 * This kernel is invoked by CharmmGBSWForce to calculate the forces acting on the system.
 */
class ReferenceCalcCharmmGBSWForceKernel : public CalcCharmmGBSWForceKernel {
public:
    ReferenceCalcCharmmGBSWForceKernel(std::string name, const Platform& platform) : CalcCharmmGBSWForceKernel(name, platform) {
    }
    ~ReferenceCalcCharmmGBSWForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the CharmmGBSWForce this kernel will be used for
     */
    void initialize(const System& system, const CharmmGBSWForce& force);
    /**  
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**  
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the CharmmGBSWForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const CharmmGBSWForce& force);
private:
    CharmmReferenceGBSW* gbsw; 
    std::vector<double> charges;
    bool isPeriodic;
};

} // namespace OpenMM

#endif // _CharmmReferenceKernels___
