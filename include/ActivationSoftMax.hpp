#ifndef ACTIVATIONSOFTMAX_HPP
#define ACTIVATIONSOFTMAX_HPP

/**************************************************************************
 * The ActivationSoftMax class implements the SoftMax activation function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

#include <algorithm>
#include <cmath>

template <unsigned int numInputs>
class ActivationSoftMax
{
	public:
		ActivationSoftMax();

        template <unsigned int numBatches>
		Matrix<numBatches, numInputs> forwardPass (const Matrix<numBatches, numInputs>& in);

    private:
};

template <unsigned int numInputs>
ActivationSoftMax<numInputs>::ActivationSoftMax()
{
}

template <unsigned int numInputs>
template <unsigned int numBatches>
Matrix<numBatches, numInputs> ActivationSoftMax<numInputs>::forwardPass (const Matrix<numBatches, numInputs>& in)
{
	Matrix<numBatches, numInputs> matOut = in;

    float maxVal = 0.0f;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int input = 0; input < numInputs; input++ )
        {
            maxVal = std::max( maxVal, matOut.at(batch, input) );
        }
    }

    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        float sum = 0.0f;
        for ( unsigned int input = 0; input < numInputs; input++ )
        {
            matOut.at( batch, input ) = exp( matOut.at(batch, input) - maxVal );
            sum += matOut.at( batch, input );
        }

        const float oneOverSum = 1.0f / sum;
        for ( unsigned int input = 0; input < numInputs; input++ )
        {
            matOut.at( batch, input ) = matOut.at( batch, input ) * oneOverSum;
        }
    }

	return matOut;
}

#endif // ACTIVATIONSOFTMAX_HPP