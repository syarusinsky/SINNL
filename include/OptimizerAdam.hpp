#ifndef OPTIMIZERADAM_HPP
#define OPTIMIZERADAM_HPP

/**************************************************************************
 * The OptimizerAdam class implements optimization using adaptive
 * momentum.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "LayerDense.hpp"

class OptimizerAdam
{
    public:
        OptimizerAdam (float learningRate, float decayRate = 0.0f, float epsilon = 1e-7f, float beta1 = 0.9f, float beta2 = 0.999f) : 
            m_LearningRate( learningRate ),
            m_DecayRate( decayRate ),
            m_DecayedLearningRate( learningRate ),
            m_Iterations( 0 ),
            m_Epsilon( epsilon ),
            m_Beta1( beta1 ),
            m_Beta2( beta2 ) {}

        template<unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
        void updateLayer (LayerDense<numBatches, numInputs, numNeurons>& layer);

        void decay();
        void resetDecay(); // resets decay related variables to starting values

    private:
        float           m_LearningRate;
        float           m_DecayRate;

        float           m_DecayedLearningRate;
        unsigned int    m_Iterations;

        float           m_Epsilon;
        float           m_Beta1;
        float           m_Beta2;
};

template<unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
void OptimizerAdam::updateLayer (LayerDense<numBatches, numInputs, numNeurons>& layer)
{
    layer.updateLayerAdam( m_DecayedLearningRate, m_Iterations, m_Epsilon, m_Beta1, m_Beta2 );
}

#endif // OPTIMIZERADAM_HPP