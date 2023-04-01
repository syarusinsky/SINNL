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
        OptimizerStochasticGradientDescent (float learningRate) : m_LearningRate( learningRate ) {}

        template<unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
        void updateLayer (LayerDense<numBatches, numInputs, numNeurons>& layer);

    private:
        float m_LearningRate;
};

template<unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
void OptimizerStochasticGradientDescent::updateLayer (LayerDense<numBatches, numInputs, numNeurons>& layer)
{
    layer.updateLayer( m_LearningRate );
}

#endif // OPTIMIZERSTOCHASTICGRADIENTDESCENT_HPP