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

#include "ReferenceNeighborList.h"
#include "CharmmReferenceKernels.h"
#include "CharmmReferenceGBMV.h"
#include "CharmmReferenceGBSW.h"
#include "ReferenceObc.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <iostream>

#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

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

// ***************************************************************************

ReferenceCalcCharmmGBMVForceKernel::~ReferenceCalcCharmmGBMVForceKernel() {
    if(gbmv) delete gbmv;
}

void ReferenceCalcCharmmGBMVForceKernel::initialize(const System& system, const CharmmGBMVForce& force) {
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
    gbmv = new CharmmReferenceGBMV(numParticles);
    gbmv->setAtomicRadii(atomicRadii);
    gbmv->setScaledRadiusFactors(scaleFactors);
    gbmv->setSolventDielectric(force.getSolventDielectric());
    gbmv->setSoluteDielectric(force.getSoluteDielectric());
    if (force.getNonbondedMethod() != CharmmGBMVForce::NoCutoff){
        OpenMM::NeighborList* neighborList = new NeighborList();
        gbmv->setUseCutoff(force.getCutoffDistance());
        gbmv->setNeighborList(*neighborList);
        if(force.getNonbondedMethod() == CharmmGBSWForce::CutoffPeriodic)
            gbmv->setPeriodic();
    }
    else {
        OpenMM::NeighborList* neighborList = NULL;
        gbmv->setNeighborList(*neighborList);
        gbmv->setNoCutoff();
        gbmv->setNoPeriodic();
    }
}

double ReferenceCalcCharmmGBMVForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    if (gbmv->getUseCutoff() && gbmv->getPeriodic())
        gbmv->setPeriodic(extractBoxVectors(context));
    if (gbmv->getUseCutoff()) {
        vector<set<int> > empty(context.getSystem().getNumParticles()); // Don't omit exclusions from the neighbor list
        OpenMM::NeighborList* neighborList = gbmv->getNeighborList();
        computeNeighborListVoxelHash(*neighborList, context.getSystem().getNumParticles(), posData, empty, extractBoxVectors(context), gbmv->getPeriodic(), gbmv->getCutoffDistance(), 0.0);
    }
    return gbmv->computeEnergyForces(posData, charges, forceData);
}

void ReferenceCalcCharmmGBMVForceKernel::copyParametersToContext(ContextImpl& context, const CharmmGBMVForce& force) {
    int numParticles = force.getNumParticles();
    if (numParticles != gbmv->getAtomicRadii().size())
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
    gbmv->setAtomicRadii(atomicRadii);
    gbmv->setScaledRadiusFactors(scaleFactors);
    return;
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

