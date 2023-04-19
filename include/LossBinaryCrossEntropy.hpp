#ifndef LOSSBINARYCROSSENTROPY_HPP
#define LOSSBINARYCROSSENTROPY_HPP

/**************************************************************************
 * The LossBinaryCrossEntroy class implements the Binary Cross-
 * Entropy loss function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "LossBase.hpp"
#include "ActivationSigmoid.hpp"
#include "Matrix.hpp"

#include <cmath>
#include <limits>

template <unsigned int numBatches, unsigned int numOutputs>
class LossBinaryCrossEntropy : public LossBase<numBatches, numOutputs>
{
    public:
        LossBinaryCrossEntropy() : m_InputsGradient() {}

        float getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const override;

        float getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const override;

        void backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, numOutputs>& targets) override;

        Matrix<numBatches, numOutputs> getInputsGradient() const { return m_InputsGradient; }

    private:
        Matrix<numBatches, numOutputs>  m_InputsGradient;
};

template <unsigned int numBatches, unsigned int numOutputs>
float LossBinaryCrossEntropy<numBatches, numOutputs>::getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const
{
    float sum = 0.0f;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        float batchSum = 0.0f;
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            const float clippedVal = std::min( 1.0f - 1e-7f, std::max(outputs.at(batch, output), 1e-7f) );
            const float lossVal = -1.0f * ( ((targets.at(batch, output) * std::log(clippedVal))) + ((1.0f - targets.at(batch, output)) * std::log(1.0f - clippedVal)) );
            batchSum += lossVal;
        }

        sum += ( batchSum ) * ( 1.0f / static_cast<float>(numOutputs) );
    }

    return sum * ( 1.0f / static_cast<float>(numBatches) );
}

template <unsigned int numBatches, unsigned int numOutputs>
float LossBinaryCrossEntropy<numBatches, numOutputs>::getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const
{
    float sum = 0.0f;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            const float outputVal = ( outputs.at(batch, output) > 0.5f ) ? 1.0f : 0.0f;
            if ( outputVal == targets.at(batch, output) )
            {
                sum += 1.0f;
            }
        }
    }

    return sum * ( 1.0f / (static_cast<float>(numBatches) * static_cast<float>(numOutputs)) );
}

template <unsigned int numBatches, unsigned int numOutputs>
void LossBinaryCrossEntropy<numBatches, numOutputs>::backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, numOutputs>& targets)
{
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            const float clippedVal = std::min( 1.0f - 1e-7f, std::max(gradient.at(batch, output), 1e-7f) );
            m_InputsGradient.at( batch, output ) = ( -1.0f * ((targets.at(batch, output) / clippedVal) - ((1.0f - targets.at(batch, output)) / (1.0f - clippedVal))) )
                                                    / static_cast<float>( numOutputs );
        }
    }

    m_InputsGradient *= ( 1.0f / static_cast<float>(numBatches) );
}

#endif // LOSSBINARYCROSSENTROPY_HPP