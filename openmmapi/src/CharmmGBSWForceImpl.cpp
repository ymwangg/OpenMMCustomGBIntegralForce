/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/OpenMMException.h"
#include "openmm/internal/CharmmGBSWForceImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/charmmKernels.h"
#include <vector>

using namespace OpenMM;
using std::vector;

CharmmGBSWForceImpl::CharmmGBSWForceImpl(const CharmmGBSWForce& owner) : owner(owner) {
}

void CharmmGBSWForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcCharmmGBSWForceKernel::Name(), context);
    if (owner.getNumParticles() != context.getSystem().getNumParticles())
        throw OpenMMException("CharmmGBSWForce must have exactly as many particles as the System it belongs to.");
    if (owner.getNonbondedMethod() == CharmmGBSWForce::CutoffPeriodic) {
        Vec3 boxVectors[3];
        context.getSystem().getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("CharmmGBSWForce: The cutoff distance cannot be greater than half the periodic box size.");
    }
    kernel.getAs<CalcCharmmGBSWForceKernel>().initialize(context.getSystem(), owner);
}

double CharmmGBSWForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcCharmmGBSWForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> CharmmGBSWForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcCharmmGBSWForceKernel::Name());
    return names;
}

void CharmmGBSWForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcCharmmGBSWForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}
