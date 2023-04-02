#ifndef LOSSCATEGORICALCROSSENTROPY_HPP
#define LOSSCATEGORICALCROSSENTROPY_HPP

/**************************************************************************
 * The LossCategoricalCrossEntroy class implements the Categorical Cross-
 * Entropy loss function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "ActivationSoftMax.hpp"
#include "Matrix.hpp"

#include <cmath>
#include <limits>

template <unsigned int numBatches, unsigned int numOutputs>
Matrix<numBatches, numOutputs> getOneHotEncodedFromScalarTargets (const Matrix<numBatches, 1>& targets)
{
    Matrix<numBatches, numOutputs> oneHotEncoded;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            oneHotEncoded.at( batch, output ) = ( targets.at(batch, 0) == static_cast<float>(output) ) ? 1.0f : 0.0f;
        }
    }

    return oneHotEncoded;
}

template <unsigned int numBatches, unsigned int numOutputs>
class LossCategoricalCrossEntropy
{
    public:
        LossCategoricalCrossEntropy() : m_InputsGradient() {}

        // for one-hot encoded targets
        float getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const;
        // for scalar targets
        float getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, 1>& targets) const;

        // for one-hot encoded targets
        float getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const;
        // for scalar targets
        float getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, 1>& targets) const;

        // for one-hot encoded targets
        void backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, numOutputs>& targets);
        // for scalar targets
        void backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, 1>& targets);

        Matrix<numBatches, numOutputs> getInputsGradient() const { return m_InputsGradient; }

    private:
        Matrix<numBatches, numOutputs>  m_InputsGradient;
};

// much faster
template <unsigned int numBatches, unsigned int numOutputs>
class LossCategoricalCrossEntropyWithActivationSoftMax
{
    public:
        LossCategoricalCrossEntropyWithActivationSoftMax() : m_LossCategoricalCrossEntropy(), m_ActivationSoftMax(), m_InputsGradient() {}

        // for one-hot encoded targets
        float getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const { return m_LossCategoricalCrossEntropy.getLoss(outputs, targets); }
        // for scalar targets
        float getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, 1>& targets) const { return m_LossCategoricalCrossEntropy.getLoss(outputs, targets); }

        // for one-hot encoded targets
        float getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const { return m_LossCategoricalCrossEntropy.getAccuracy(outputs, targets); }
        // for scalar targets
        float getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, 1>& targets) const { return m_LossCategoricalCrossEntropy.getAccuracy(outputs, targets); }

        Matrix<numBatches, numOutputs> forwardPass (const Matrix<numBatches, numOutputs>& in) const { return m_ActivationSoftMax.forwardPass( in ); }

        // for one-hot encoded targets
        void backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, numOutputs>& targets);
        // for scalar targets
        void backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, 1>& targets) { backwardPass(gradient, getOneHotEncodedFromScalarTargets<numBatches, numOutputs>(targets)); }

        Matrix<numBatches, numOutputs> getInputsGradient() const { return m_InputsGradient; }

    private:
        LossCategoricalCrossEntropy<numBatches, numOutputs>     m_LossCategoricalCrossEntropy;
        ActivationSoftMax<numBatches, numOutputs>               m_ActivationSoftMax;

        Matrix<numBatches, numOutputs>                          m_InputsGradient;
};

template <unsigned int numBatches, unsigned int numOutputs>
float LossCategoricalCrossEntropy<numBatches, numOutputs>::getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const
{
    Matrix<numOutputs, numBatches> targetsTransposed = targets.transpose();
    Matrix<numBatches, numBatches> result = matrixDotProduct<numBatches, numOutputs, numOutputs, numBatches>( outputs, targetsTransposed );

    float sum = 0.0f;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        const float resultVal = result.at( batch, batch );
        sum += -1.0f * log( std::max(0.0f + std::numeric_limits<float>::min(), std::min(resultVal, std::numeric_limits<float>::max())) );
    }

    return sum * ( 1.0f / numBatches );
}


template <unsigned int numBatches, unsigned int numOutputs>
float LossCategoricalCrossEntropy<numBatches, numOutputs>::getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, 1>& targets) const
{
    return getLoss( outputs, getOneHotEncodedFromScalarTargets<numBatches, numOutputs>(targets) );
}

template <unsigned int numBatches, unsigned int numOutputs>
float LossCategoricalCrossEntropy<numBatches, numOutputs>::getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const
{
    float sum = 0.0f;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        unsigned int index = 0;
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            if ( outputs.at(batch, output) > outputs.at(batch, index) )
            {
                index = output;
            }
        }

        sum += ( targets.at(batch, index) == 1.0f ) ? 1.0f : 0.0f;
    }

    return sum * ( 1.0f / numBatches );
}

template <unsigned int numBatches, unsigned int numOutputs>
float LossCategoricalCrossEntropy<numBatches, numOutputs>::getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, 1>& targets) const
{
    Matrix<numBatches, numOutputs> oneHot = getOneHotEncodedFromScalarTargets<numBatches, numOutputs>( targets );

    return getAccuracy( outputs, oneHot );
}

template <unsigned int numBatches, unsigned int numOutputs>
void LossCategoricalCrossEntropy<numBatches, numOutputs>::backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, numOutputs>& targets)
{
    m_InputsGradient = ( targets * -1.0f );
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            m_InputsGradient.at( batch, output ) /= gradient.at( batch, output );
        }
    }
    m_InputsGradient *= ( 1.0f / static_cast<float>(numBatches) );
}

template <unsigned int numBatches, unsigned int numOutputs>
void LossCategoricalCrossEntropy<numBatches, numOutputs>::backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, 1>& targets)
{
    backwardPass( gradient, getOneHotEncodedFromScalarTargets<numBatches, numOutputs>(targets) );
}

template <unsigned int numBatches, unsigned int numOutputs>
void LossCategoricalCrossEntropyWithActivationSoftMax<numBatches, numOutputs>::backwardPass(const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, numOutputs>& targets)
{
    m_InputsGradient = gradient;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            m_InputsGradient.at( batch, output ) -= ( 1.0f * targets.at(batch, output) );
            m_InputsGradient.at( batch, output ) *= ( 1.0f / numBatches );
        }
    }
}

#endif // LOSSCATEGORICALCROSSENTROPY_HPP