#ifndef LOSSCATEGORICALCROSSENTROPY_HPP
#define LOSSCATEGORICALCROSSENTROPY_HPP

/**************************************************************************
 * The LossCategoricalCrossEntroy class implements the Categorical Cross-
 * Entropy loss function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

#include <cmath>
#include <limits>

template <unsigned int numBatches, unsigned int numOutputs>
class LossCategoricalCrossEntropy
{
    public:
        LossCategoricalCrossEntropy() {}

        // for one-hot encoded targets
        float getLoss(const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets);
        // for scalar targets
        float getLoss(const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, 1>& targets);
};

template <unsigned int numBatches, unsigned int numOutputs>
float LossCategoricalCrossEntropy<numBatches, numOutputs>::getLoss(const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets)
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
float LossCategoricalCrossEntropy<numBatches, numOutputs>::getLoss(const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, 1>& targets)
{
    Matrix<numBatches, numOutputs> oneHotEncoded;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int output = 0; output < numOutputs; output++ )
        {
            oneHotEncoded.at( batch, output ) = ( targets.at(batch, 0) == static_cast<float>(output) ) ? 1.0f : 0.0f;
        }
    }

    return getLoss( outputs, oneHotEncoded );
}

#endif // LOSSCATEGORICALCROSSENTROPY_HPP