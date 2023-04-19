#ifndef LOSSMEANSQUAREDERROR_HPP
#define LOSSMEANSQUAREDERROR_HPP

/**************************************************************************
 * The LossMeanSquaredError class implements the Mean Squared
 * Error loss function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "LossBase.hpp"
#include "Matrix.hpp"

#include <cmath>
#include <limits>

template <unsigned int numBatches, unsigned int numOutputs>
class LossMeanSquaredError : public LossBase<numBatches, numOutputs>
{
    public:
        LossMeanSquaredError() : m_InputsGradient() {}

        float getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const override;

        float getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const override;

        void backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, numOutputs>& targets) override;

        Matrix<numBatches, numOutputs> getInputsGradient() const { return m_InputsGradient; }

    private:
        Matrix<numBatches, numOutputs>  m_InputsGradient;
};

template <unsigned int numBatches, unsigned int numOutputs>
float LossMeanSquaredError<numBatches, numOutputs>::getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const
{
    float sum = 0.0f;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        float batchSum = 0.0f;
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            batchSum += std::pow( targets.at(batch, output) - outputs.at(batch, output), 2 );
        }

        sum += ( batchSum ) * ( 1.0f / static_cast<float>(numOutputs) );
    }

    return sum * ( 1.0f / static_cast<float>(numBatches) );
}

template <unsigned int numBatches, unsigned int numOutputs>
float LossMeanSquaredError<numBatches, numOutputs>::getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const
{
    float mean = 0.0f;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            mean += targets.at( batch, output );
        }
    }
    mean *= ( 1.0f / (numOutputs * numBatches) );

    float squaredSum = 0.0f;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            squaredSum += std::pow( targets.at(batch, output) - mean, 2 );
        }
    }

    const float accuracyPrecision = std::sqrt( squaredSum * (1.0f / (numOutputs * numBatches)) ) * ( 1.0f / 250.0f );

    float sum = 0.0f;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            sum += ( std::abs(outputs.at(batch, output) - targets.at(batch, output)) < accuracyPrecision ) ? 1.0f : 0.0f;
        }
    }

    return sum * ( 1.0f / (static_cast<float>(numBatches) * static_cast<float>(numOutputs)) );
}

template <unsigned int numBatches, unsigned int numOutputs>
void LossMeanSquaredError<numBatches, numOutputs>::backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, numOutputs>& targets)
{
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            const float val = -2.0f * ( (targets.at(batch, output) - gradient.at(batch, output)) / numOutputs );
            m_InputsGradient.at( batch, output ) = val;
        }
    }

    m_InputsGradient *= ( 1.0f / static_cast<float>(numBatches) );
}

#endif // LOSSMEANSQUAREDERROR_HPP