#ifndef LOSSBASE_HPP
#define LOSSBASE_HPP

/**************************************************************************
 * The LossBase class is the base class for all loss functions.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

template <unsigned int numBatches, unsigned int numOutputs>
class LossBase
{
    public:
        LossBase() {}
        virtual ~LossBase() {}

        virtual float getLoss (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const = 0;

        virtual float getAccuracy (const Matrix<numBatches, numOutputs>& outputs, const Matrix<numBatches, numOutputs>& targets) const = 0;

        virtual void backwardPass (const Matrix<numBatches, numOutputs>& gradient, const Matrix<numBatches, numOutputs>& targets) = 0;
};

#endif // LOSSCATEGORICALCROSSENTROPY_HPP