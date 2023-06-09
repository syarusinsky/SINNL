#include "OptimizerStochasticGradientDescent.hpp"

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