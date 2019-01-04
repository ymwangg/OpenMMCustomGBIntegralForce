#ifndef OPENMM_CHARMM_GBMV_FORCE_H_
#define OPENMM_CHARMM_GBMV_FORCE_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMCharmm                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs, Peter Eastman                                    *
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

#include "openmm/Force.h"
#include <vector>
#include "internal/windowsExportCharmm.h"
#include "TabulatedFunction.h"
#include "Vec3.h"
#include <map>
#include <set>
#include <utility>

namespace OpenMM {
/**
 * This class implements an implicit solvation force using the GBSA-OBC model.
 *
 * To use this class, create a GBSAOBCForce object, then call addParticle() once for each particle in the
 * System to define its parameters.  The number of particles for which you define GBSA parameters must
 * be exactly equal to the number of particles in the System, or else an exception will be thrown when you
 * try to create a Context.  After a particle has been added, you can modify its force field parameters
 * by calling setParticleParameters().  This will have no effect on Contexts that already exist unless you
 * call updateParametersInContext().
 *
 * When using this Force, the System should also include a NonbondedForce, and both objects must specify
 * identical charges for all particles.  Otherwise, the results will not be correct.  Furthermore, if the
 * nonbonded method is set to CutoffNonPeriodic or CutoffPeriodic, you should call setReactionFieldDielectric(1.0)
 * on the NonbondedForce to turn off the reaction field approximation, which does not produce correct results
 * when combined with GBSA.
 */


class OPENMM_EXPORT_CHARMM CharmmGBMVForce : public Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Interactions beyond the cutoff distance are ignored.
         */
        CutoffNonPeriodic = 1,
        /**
         * Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of
         * each other particle.  Interactions beyond the cutoff distance are ignored.
         */
        CutoffPeriodic = 2,
    };
    /**
     * This is an enumeration of the different ways in which a computed value or energy term can be calculated.
     */
    enum ComputationType {
        /**
         * The value is computed independently for each particle, based only on the parameters and computed values for that particle.
         */
        SingleParticle = 0,
        /**
         * The value is computed as a sum over all pairs of particles, except those which have been added as exclusions.
         */
        ParticlePair = 1,
        ParticlePairNoExclusions = 2,
        VolumeIntegral = 3,
    };
    /**
     * Create a CharmmGBMVForce.
     */
    CharmmGBMVForce();
    ~CharmmGBMVForce();
    /**
     * Get the number of particles for which force field parameters have been defined.
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Get the number of particle pairs whose interactions should be excluded.
     */
    int getNumExclusions() const {
        return exclusions.size();
    }
    /**
     * Get the number of per-particle parameters that the interaction depends on.
     */
    int getNumPerParticleParameters() const {
        return parameters.size();
    }
    /**
     * Get the number of global parameters that the interaction depends on.
     */
    int getNumGlobalParameters() const {
        return globalParameters.size();
    }
    /**
     * Get the number of global parameters with respect to which the derivative of the energy
     * should be computed.
     */
    int getNumEnergyParameterDerivatives() const {
        return energyParameterDerivatives.size();
    }
    /**
     * Get the number of tabulated functions that have been defined.
     */
    int getNumTabulatedFunctions() const {
        return functions.size();
    }
    /**
     * Get the number of tabulated functions that have been defined.
     *
     * @deprecated This method exists only for backward compatibility.  Use getNumTabulatedFunctions() instead.
     */
    int getNumFunctions() const {
        return functions.size();
    }
    /**
     * Get the number of per-particle computed values the interaction depends on.
     */
    int getNumComputedValues() const {
        return computedValues.size();
    }

    int getNumVolumeIntegrals() const {
        return computedVolumeIntegrals.size();
    }
    /**
     * Get the number of terms in the energy computation.
     */
    int getNumEnergyTerms() const {
        return energyTerms.size();
    }
    /**
     * Get the method used for handling long range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);
    /**
     * Add a new per-particle parameter that the interaction may depend on.
     *
     * @param name     the name of the parameter
     * @return the index of the parameter that was added
     */
    int addPerParticleParameter(const std::string& name);
    /**
     * Get the name of a per-particle parameter.
     *
     * @param index     the index of the parameter for which to get the name
     * @return the parameter name
     */
    const std::string& getPerParticleParameterName(int index) const;
    /**
     * Set the name of a per-particle parameter.
     *
     * @param index          the index of the parameter for which to set the name
     * @param name           the name of the parameter
     */
    void setPerParticleParameterName(int index, const std::string& name);
    /**
     * Add a new global parameter that the interaction may depend on.  The default value provided to
     * this method is the initial value of the parameter in newly created Contexts.  You can change
     * the value at any time by calling setParameter() on the Context.
     *
     * @param name             the name of the parameter
     * @param defaultValue     the default value of the parameter
     * @return the index of the parameter that was added
     */
    int addGlobalParameter(const std::string& name, double defaultValue);
    /**
     * Get the name of a global parameter.
     *
     * @param index     the index of the parameter for which to get the name
     * @return the parameter name
     */
    const std::string& getGlobalParameterName(int index) const;
    /**
     * Set the name of a global parameter.
     *
     * @param index          the index of the parameter for which to set the name
     * @param name           the name of the parameter
     */
    void setGlobalParameterName(int index, const std::string& name);
    /**
     * Get the default value of a global parameter.
     *
     * @param index     the index of the parameter for which to get the default value
     * @return the parameter default value
     */
    double getGlobalParameterDefaultValue(int index) const;
    /**
     * Set the default value of a global parameter.
     *
     * @param index         the index of the parameter for which to set the default value
     * @param defaultValue  the default value of the parameter
     */
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    /**
     * Request that this Force compute the derivative of its energy with respect to a global parameter.
     * The parameter must have already been added with addGlobalParameter().
     *
     * @param name             the name of the parameter
     */
    void addEnergyParameterDerivative(const std::string& name);
    /**
     * Get the name of a global parameter with respect to which this Force should compute the
     * derivative of the energy.
     *
     * @param index     the index of the parameter derivative, between 0 and getNumEnergyParameterDerivatives()
     * @return the parameter name
     */
    const std::string& getEnergyParameterDerivativeName(int index) const;
    /**
     * Add the nonbonded force parameters for a particle.  This should be called once for each particle
     * in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.
     *
     * @param parameters    the list of parameters for the new particle
     * @return the index of the particle that was added
     */
    int addParticle(const std::vector<double>& parameters=std::vector<double>());
    /**
     * Get the nonbonded force parameters for a particle.
     *
     * @param index            the index of the particle for which to get parameters
     * @param[out] parameters  the list of parameters for the specified particle
     */
    void getParticleParameters(int index, std::vector<double>& parameters) const;
    /**
     * Set the nonbonded force parameters for a particle.
     *
     * @param index       the index of the particle for which to set parameters
     * @param parameters  the list of parameters for the specified particle
     */
    void setParticleParameters(int index, const std::vector<double>& parameters);
    /**
     * Add a computed value to calculate for each particle.
     *
     * @param name        the name of the value
     * @param expression  an algebraic expression to evaluate when calculating the computed value.  If the
     *                    ComputationType is SingleParticle, the expression is evaluated independently
     *                    for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
     *                    parameters and previous computed values for that particle.  If the ComputationType is ParticlePair
     *                    or ParticlePairNoExclusions, the expression is evaluated once for every other
     *                    particle in the system and summed to get the final value.  In the latter case,
     *                    the expression may depend on the distance r between the two particles, and on
     *                    the per-particle parameters and previous computed values for each of them.
     *                    Append "1" to a variable name to indicate the parameter for the particle whose
     *                    value is being calculated, and "2" to indicate the particle it is interacting with.
     * @param type        the method to use for computing this value
     */
    int addComputedValue(const std::string& name, const std::string& expression, ComputationType type);
    /**
     * Get the properties of a computed value.
     *
     * @param index            the index of the computed value for which to get parameters
     * @param[out] name        the name of the value
     * @param[out] expression  an algebraic expression to evaluate when calculating the computed value.  If the
     *                         ComputationType is SingleParticle, the expression is evaluated independently
     *                         for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
     *                         parameters and previous computed values for that particle.  If the ComputationType is ParticlePair
     *                         or ParticlePairNoExclusions, the expression is evaluated once for every other
     *                         particle in the system and summed to get the final value.  In the latter case,
     *                         the expression may depend on the distance r between the two particles, and on
     *                         the per-particle parameters and previous computed values for each of them.
     *                         Append "1" to a variable name to indicate the parameter for the particle whose
     *                         value is being calculated, and "2" to indicate the particle it is interacting with.
     * @param[out] type        the method to use for computing this value
     */
    int addVolumeIntegral(const std::string& name, const std::map<std::string, double> parameters);
    void getComputedValueParameters(int index, std::string& name, std::string& expression, ComputationType& type) const;
    void getVolumeIntegralParameters(int index, std::string& name, std::map<std::string, double> parameters) const;
    /**
     * Set the properties of a computed value.
     *
     * @param index       the index of the computed value for which to set parameters
     * @param name        the name of the value
     * @param expression  an algebraic expression to evaluate when calculating the computed value.  If the
     *                    ComputationType is SingleParticle, the expression is evaluated independently
     *                    for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
     *                    parameters and previous computed values for that particle.  If the ComputationType is ParticlePair
     *                    or ParticlePairNoExclusions, the expression is evaluated once for every other
     *                    particle in the system and summed to get the final value.  In the latter case,
     *                    the expression may depend on the distance r between the two particles, and on
     *                    the per-particle parameters and previous computed values for each of them.
     *                    Append "1" to a variable name to indicate the parameter for the particle whose
     *                    value is being calculated, and "2" to indicate the particle it is interacting with.
     * @param type        the method to use for computing this value
     */
    void setComputedValueParameters(int index, const std::string& name, const std::string& expression, ComputationType type);
    void setVolumeIntegralParameters(int index, const std::string& name, const std::map<std::string, double> parameters);
    /**
     * Add a term to the energy computation.
     *
     * @param expression  an algebraic expression to evaluate when calculating the energy.  If the
     *                    ComputationType is SingleParticle, the expression is evaluated once
     *                    for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
     *                    parameters and computed values for that particle.  If the ComputationType is ParticlePair or
     *                    ParticlePairNoExclusions, the expression is evaluated once for every pair of
     *                    particles in the system.  In the latter case,
     *                    the expression may depend on the distance r between the two particles, and on
     *                    the per-particle parameters and computed values for each of them.
     *                    Append "1" to a variable name to indicate the parameter for the first particle
     *                    in the pair and "2" to indicate the second particle in the pair.
     * @param type        the method to use for computing this value
     */
    int addEnergyTerm(const std::string& expression, ComputationType type);
    /**
     * Get the properties of a term to the energy computation.
     *
     * @param index            the index of the term for which to get parameters
     * @param[out] expression  an algebraic expression to evaluate when calculating the energy.  If the
     *                         ComputationType is SingleParticle, the expression is evaluated once
     *                         for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
     *                         parameters and computed values for that particle.  If the ComputationType is ParticlePair or
     *                         ParticlePairNoExclusions, the expression is evaluated once for every pair of
     *                         particles in the system.  In the latter case,
     *                         the expression may depend on the distance r between the two particles, and on
     *                         the per-particle parameters and computed values for each of them.
     *                         Append "1" to a variable name to indicate the parameter for the first particle
     *                         in the pair and "2" to indicate the second particle in the pair.
     * @param[out] type        the method to use for computing this value
     */
    void getEnergyTermParameters(int index, std::string& expression, ComputationType& type) const;
    /**
     * Set the properties of a term to the energy computation.
     *
     * @param index       the index of the term for which to set parameters
     * @param expression  an algebraic expression to evaluate when calculating the energy.  If the
     *                    ComputationType is SingleParticle, the expression is evaluated once
     *                    for each particle, and may depend on its x, y, and z coordinates, as well as the per-particle
     *                    parameters and computed values for that particle.  If the ComputationType is ParticlePair or
     *                    ParticlePairNoExclusions, the expression is evaluated once for every pair of
     *                    particles in the system.  In the latter case,
     *                    the expression may depend on the distance r between the two particles, and on
     *                    the per-particle parameters and computed values for each of them.
     *                    Append "1" to a variable name to indicate the parameter for the first particle
     *                    in the pair and "2" to indicate the second particle in the pair.
     * @param type        the method to use for computing this value
     */
    void setEnergyTermParameters(int index, const std::string& expression, ComputationType type);
    /**
     * Add a particle pair to the list of interactions that should be excluded.
     *
     * @param particle1  the index of the first particle in the pair
     * @param particle2  the index of the second particle in the pair
     * @return the index of the exclusion that was added
     */
    int addExclusion(int particle1, int particle2);
    /**
     * Get the particles in a pair whose interaction should be excluded.
     *
     * @param index           the index of the exclusion for which to get particle indices
     * @param[out] particle1  the index of the first particle in the pair
     * @param[out] particle2  the index of the second particle in the pair
     */
    void getExclusionParticles(int index, int& particle1, int& particle2) const;
    /**
     * Set the particles in a pair whose interaction should be excluded.
     *
     * @param index      the index of the exclusion for which to set particle indices
     * @param particle1  the index of the first particle in the pair
     * @param particle2  the index of the second particle in the pair
     */
    void setExclusionParticles(int index, int particle1, int particle2);
    /**
     * Add a tabulated function that may appear in expressions.
     *
     * @param name           the name of the function as it appears in expressions
     * @param function       a TabulatedFunction object defining the function.  The TabulatedFunction
     *                       should have been created on the heap with the "new" operator.  The
     *                       Force takes over ownership of it, and deletes it when the Force itself is deleted.
     * @return the index of the function that was added
     */
    int addTabulatedFunction(const std::string& name, TabulatedFunction* function);
    /**
     * Get a const reference to a tabulated function that may appear in expressions.
     *
     * @param index     the index of the function to get
     * @return the TabulatedFunction object defining the function
     */
    const TabulatedFunction& getTabulatedFunction(int index) const;
    /**
     * Get a reference to a tabulated function that may appear in expressions.
     *
     * @param index     the index of the function to get
     * @return the TabulatedFunction object defining the function
     */
    TabulatedFunction& getTabulatedFunction(int index);
    /**
     * Get the name of a tabulated function that may appear in expressions.
     *
     * @param index     the index of the function to get
     * @return the name of the function as it appears in expressions
     */
    const std::string& getTabulatedFunctionName(int index) const;
    /**
     * Add a tabulated function that may appear in expressions.
     *
     * @deprecated This method exists only for backward compatibility.  Use addTabulatedFunction() instead.
     */
    int addFunction(const std::string& name, const std::vector<double>& values, double min, double max);
    /**
     * Get the parameters for a tabulated function that may appear in expressions.
     *
     * @deprecated This method exists only for backward compatibility.  Use getTabulatedFunctionParameters() instead.
     * If the specified function is not a Continuous1DFunction, this throws an exception.
     */
    void getFunctionParameters(int index, std::string& name, std::vector<double>& values, double& min, double& max) const;
    /**
     * Set the parameters for a tabulated function that may appear in expressions.
     *
     * @deprecated This method exists only for backward compatibility.  Use setTabulatedFunctionParameters() instead.
     * If the specified function is not a Continuous1DFunction, this throws an exception.
     */
    void setFunctionParameters(int index, const std::string& name, const std::vector<double>& values, double min, double max);
    /**
     * Update the per-particle parameters in a Context to match those stored in this Force object.  This method provides
     * an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setParticleParameters() to modify this object's parameters, then call updateParametersInContext()
     * to copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the values of per-particle parameters.
     * All other aspects of the Force (such as the energy function) are unaffected and can only be changed by reinitializing
     * the Context.  Also, this method cannot be used to add new particles, only to change the parameters of existing ones.
     */
    void updateParametersInContext(Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == CharmmGBMVForce::CutoffPeriodic;
    }
protected:
    ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    class PerParticleParameterInfo;
    class GlobalParameterInfo;
    class ExclusionInfo;
    class FunctionInfo;
    class ComputationInfo;
    class VolumeIntegralInfo;
    NonbondedMethod nonbondedMethod;
    double cutoffDistance;
    std::vector<PerParticleParameterInfo> parameters;
    std::vector<GlobalParameterInfo> globalParameters;
    std::vector<ParticleInfo> particles;
    std::vector<ExclusionInfo> exclusions;
    std::vector<FunctionInfo> functions;
    std::vector<ComputationInfo> computedValues;
    std::vector<VolumeIntegralInfo> computedVolumeIntegrals;
    std::vector<ComputationInfo> energyTerms;
    std::vector<int> energyParameterDerivatives;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class CharmmGBMVForce::ParticleInfo {
public:
    std::vector<double> parameters;
    ParticleInfo() {
    }
    ParticleInfo(const std::vector<double>& parameters) : parameters(parameters) {
    }
};
/**
 * This is an internal class used to record information about a per-particle parameter.
 * @private
 */
class CharmmGBMVForce::PerParticleParameterInfo {
public:
    std::string name;
    PerParticleParameterInfo() {
    }
    PerParticleParameterInfo(const std::string& name) : name(name) {
    }
};
/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class CharmmGBMVForce::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};

/**
 * This is an internal class used to record information about an exclusion.
 * @private
 */
class CharmmGBMVForce::ExclusionInfo {
public:
    int particle1, particle2;
    ExclusionInfo() {
        particle1 = particle2 = -1;
    }
    ExclusionInfo(int particle1, int particle2) :
        particle1(particle1), particle2(particle2) {
    }
};

/**
 * This is an internal class used to record information about a tabulated function.
 * @private
 */
class CharmmGBMVForce::FunctionInfo {
public:
    std::string name;
    TabulatedFunction* function;
    FunctionInfo() {
    }
    FunctionInfo(const std::string& name, TabulatedFunction* function) : name(name), function(function) {
    }
};

/**
 * This is an internal class used to record information about a computed value or energy term.
 * @private
 */
class CharmmGBMVForce::ComputationInfo {
public:
    std::string name;
    std::string expression;
    CharmmGBMVForce::ComputationType type;
    ComputationInfo() {
    }
    ComputationInfo(const std::string& name, const std::string& expression, CharmmGBMVForce::ComputationType type) :
        name(name), expression(expression), type(type) {
    }
};

class CharmmGBMVForce::VolumeIntegralInfo {
public:
    std::string name;
    std::map<std::string, double> parameters;
    VolumeIntegralInfo() {
    }
    VolumeIntegralInfo(const std::string& name, const std::map<std::string, double> parameter) :
        name(name), parameters(parameters) {
    }
};


} // namespace OpenMM

#endif /*OPENMM_CHARMM_GBMV_FORCE_H_*/
