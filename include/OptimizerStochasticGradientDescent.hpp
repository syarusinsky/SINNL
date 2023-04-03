#ifndef OPTIMIZERSTOCHASTICGRADIENTDESCENT_HPP
#define OPTIMIZERSTOCHASTICGRADIENTDESCENT_HPP

/**************************************************************************
 * The OptimizerStochasticGradientDescent class implements optimization
 * using stochastic gradient descent.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "LayerDense.hpp"

class OptimizerStochasticGradientDescent
{
    public:
        OptimizerStochasticGradientDescent (float learningRate, float decayRate = 0.0f, float momentum = 0.0f) : 
            m_LearningRate( learningRate ),
            m_DecayRate( decayRate ),
            m_DecayedLearningRate( learningRate ),
            m_Iterations( 0 ),
            m_Momentum( momentum ) {}

        template<unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
        void updateLayer (LayerDense<numBatches, numInputs, numNeurons>& layer);

        void decay();
        void resetDecay(); // resets decay related variables to starting values

    private:
        float           m_LearningRate;
        float           m_DecayRate;

        float           m_DecayedLearningRate;
        unsigned int    m_Iterations;

        float           m_Momentum;
};

template<unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
void OptimizerStochasticGradientDescent::updateLayer (LayerDense<numBatches, numInputs, numNeurons>& layer)
{
    layer.updateLayer( m_DecayedLearningRate, m_Momentum );
}

void OptimizerStochasticGradientDescent::decay()
{
    m_DecayedLearningRate = m_LearningRate * ( 1.0f / (1.0f + (m_DecayRate * m_Iterations)) );
    m_Iterations++;
}

void OptimizerStochasticGradientDescent::resetDecay()
{
    m_DecayedLearningRate = m_LearningRate;
    m_Iterations = 0;
}

#endif // OPTIMIZERSTOCHASTICGRADIENTDESCENT_HPP