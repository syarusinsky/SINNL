#ifndef ACTIVATIONRELU_HPP
#define ACTIVATIONRELU_HPP

/**************************************************************************
 * The ActivationReLU class implements the ReLU activation function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

#include <algorithm>

template <unsigned int numBatches, unsigned int numInputs>
class ActivationReLU
{
	public:
		ActivationReLU();

		Matrix<numBatches, numInputs> forwardPass (const Matrix<numBatches, numInputs>& in);

        void backwardPass (const Matrix<numBatches, numInputs>& gradient);

        Matrix<numBatches, numInputs> getGradient() { return m_Gradient; }

    private:
        Matrix<numBatches, numInputs>   m_Gradient;
};

template <unsigned int numBatches, unsigned int numInputs>
ActivationReLU<numBatches, numInputs>::ActivationReLU()
{
}

template <unsigned int numBatches, unsigned int numInputs>
Matrix<numBatches, numInputs> ActivationReLU<numBatches, numInputs>::forwardPass (const Matrix<numBatches, numInputs>& in)
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

template <unsigned int numBatches, unsigned int numInputs>
void ActivationReLU<numBatches, numInputs>::backwardPass (const Matrix<numBatches, numInputs>& gradient)
{
    m_Gradient = gradient;

    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        for ( unsigned int input = 0; input < numInputs; input++ )
        {
            m_Gradient.at( batch, input ) = ( m_Gradient.at(batch, input) > 0.0f ) ? m_Gradient.at( batch, input ) : 0.0f;
        }
    }
}

#endif // ACTIVATIONRELU_HPP