#ifndef ACTIVATIONSIGMOID_HPP
#define ACTIVATIONSIGMOID_HPP

/**************************************************************************
 * The ActivationSigmoid class implements the Sigmoid activation function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

#include <algorithm>

template <unsigned int numBatches, unsigned int numInputs>
class ActivationSigmoid
{
	public:
		ActivationSigmoid();

		Matrix<numBatches, numInputs> forwardPass (const Matrix<numBatches, numInputs>& in) const;

        void backwardPass (const Matrix<numBatches, numInputs>& in, const Matrix<numBatches, numInputs>& gradient);

        Matrix<numBatches, numInputs> getInputsGradient() const { return m_Gradient; }

    private:
        Matrix<numBatches, numInputs>   m_Gradient;
};

template <unsigned int numBatches, unsigned int numInputs>
ActivationSigmoid<numBatches, numInputs>::ActivationSigmoid()
{
}

template <unsigned int numBatches, unsigned int numInputs>
Matrix<numBatches, numInputs> ActivationSigmoid<numBatches, numInputs>::forwardPass (const Matrix<numBatches, numInputs>& in) const
{
	Matrix<numBatches, numInputs> matOut = in;

    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int input = 0; input < numInputs; input++ )
        {
            matOut.at( batch, input ) = 1.0f / ( 1.0f + std::exp(matOut.at(batch, input) * -1.0f) );
        }
    }

	return matOut;
}

template <unsigned int numBatches, unsigned int numInputs>
void ActivationSigmoid<numBatches, numInputs>::backwardPass (const Matrix<numBatches, numInputs>& in, const Matrix<numBatches, numInputs>& gradient)
{
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int input = 0; input < numInputs; input++ )
        {
            m_Gradient.at( batch, input ) = gradient.at( batch, input ) * ( 1.0f - in.at(batch, input) ) * in.at( batch, input );
        }
    }
}

#endif // ACTIVATIONSIGMOID_HPP