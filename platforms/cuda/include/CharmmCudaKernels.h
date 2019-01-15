
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

#ifndef __CharmmCudaKernels_H__
#define __CharmmCudaKernels_H__

#include "CustomGBIntegral.h"
#include "openmm/charmmKernels.h"
#include "CudaPlatform.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "CudaParameterSet.h"
#include "openmm/System.h"
#include "openmm/internal/CompiledExpressionSet.h"
#include "openmm/internal/CustomIntegratorUtilities.h"
#include "lepton/CompiledExpression.h"
#include "lepton/ExpressionProgram.h"


namespace OpenMM {

/**
 * This kernel is invoked by CharmmGBMVForce to calculate the forces acting on the system.
 */
class CudaCalcCharmmGBMVForceKernel : public CalcCharmmGBMVForceKernel {
public:
    CudaCalcCharmmGBMVForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) : CalcCharmmGBMVForceKernel(name, platform), hasInitializedKernels(false), cu(cu), params(NULL), computedValues(NULL), energyDerivs(NULL), energyDerivChain(NULL), system(system) {
    }
    ~CudaCalcCharmmGBMVForceKernel();
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
private:
    CustomGBIntegral* integralMethod;
    class ForceInfo;
    double cutoff;
    bool hasInitializedKernels, needParameterGradient, needEnergyParamDerivs;
    int maxTiles, numComputedValues, numComputedIntegrals;
    int numAtoms;
    CudaContext& cu;
    ForceInfo* info;
    CudaParameterSet* params;
    CudaParameterSet* computedIntegrals;
    CudaParameterSet* computedValues;
    CudaParameterSet* energyDerivs;
    CudaParameterSet* energyDerivChain;
    std::vector<CudaParameterSet*> dValuedParam;
    CudaArray lookupTable;
    CudaArray lookupTableNumAtoms;
    
    CudaArray longEnergyDerivs;
    CudaArray globals;
    CudaArray valueBuffers;
    CudaArray integralGradients;
    std::vector<std::string> globalParamNames;
    std::vector<float> globalParamValues;
    std::vector<CudaArray> tabulatedFunctions;
    std::vector<bool> pairValueUsesParam, pairEnergyUsesParam, pairEnergyUsesValue, pairEnergyUsesIntegral;
    const System& system;
    CUfunction  lookupTableKernel, perParticleValueKernel, pairEnergyKernel, perParticleEnergyKernel, gradientChainRuleKernel;
    std::vector<void*> lookupTableArgs, pairValueArgs, perParticleValueArgs, pairEnergyArgs, perParticleEnergyArgs, gradientChainRuleArgs;
    std::string pairValueSrc, pairEnergySrc;
    std::map<std::string, std::string> pairValueDefines, pairEnergyDefines;
};

} // namespace OpenMM

#endif // _CharmmCudaKernels___
