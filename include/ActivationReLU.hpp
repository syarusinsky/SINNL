#ifndef ACTIVATIONRELU_HPP
#define ACTIVATIONRELU_HPP

/**************************************************************************
 * The ActivationReLU class implements the ReLU activation function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

#include <algorithm>

template <unsigned int numInputs>
class ActivationReLU
{
	public:
		ActivationReLU();

        template <unsigned int numBatches>
		Matrix<numBatches, numInputs> forwardPass (const Matrix<numBatches, numInputs>& in);

    private:
};

template <unsigned int numInputs>
ActivationReLU<numInputs>::ActivationReLU()
{
}

template <unsigned int numInputs>
template <unsigned int numBatches>
Matrix<numBatches, numInputs> ActivationReLU<numInputs>::forwardPass (const Matrix<numBatches, numInputs>& in)
{
	Matrix<numBatches, numInputs> matOut = in;

    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int input = 0; input < numInputs; input++ )
        {
            matOut.at( batch, input ) = std::max( 0.0f, matOut.at(batch, input) );
        }
    }

	return matOut;
}

#endif // ACTIVATIONRELU_HPP