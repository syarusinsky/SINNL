#ifndef ACTIVATIONSOFTMAX_HPP
#define ACTIVATIONSOFTMAX_HPP

/**************************************************************************
 * The ActivationSoftMax class implements the SoftMax activation function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

#include <algorithm>
#include <cmath>

template <unsigned int numBatches, unsigned int numInputs>
class ActivationSoftMax
{
	public:
		ActivationSoftMax();

		Matrix<numBatches, numInputs> forwardPass (const Matrix<numBatches, numInputs>& in) const;

        void backwardPass (const Matrix<numBatches, numInputs>& in, const Matrix<numBatches, numInputs>& gradient);

        Matrix<numBatches, numInputs> getInputsGradient() const { return m_InputsGradient; }

    private:
        Matrix<numBatches, numInputs>   m_InputsGradient;
};

template <unsigned int numBatches, unsigned int numInputs>
ActivationSoftMax<numBatches, numInputs>::ActivationSoftMax() :
    m_InputsGradient()
{
}

template <unsigned int numBatches, unsigned int numInputs>
Matrix<numBatches, numInputs> ActivationSoftMax<numBatches, numInputs>::forwardPass (const Matrix<numBatches, numInputs>& in) const
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

template <unsigned int numBatches, unsigned int numInputs>
void ActivationSoftMax<numBatches, numInputs>::backwardPass (const Matrix<numBatches, numInputs>& in, const Matrix<numBatches, numInputs>& gradient)
{
    m_InputsGradient = Matrix<numBatches, numInputs>();

    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        Matrix<1, numInputs> gradientRowVector;
        Matrix<numInputs, 1> sampleColumnVector;
        Matrix<numInputs, numInputs> sampleSquare;
        for ( unsigned int input = 0; input < numInputs; input++ )
        {
            sampleColumnVector.at( input, 0 ) = in.at( batch, input );
            sampleSquare.at( input, input ) = in.at( batch, input );
            gradientRowVector.at( 0, input ) = gradient.at( batch, input );
        }

        Matrix<numInputs, numInputs> jacobian = sampleSquare - matrixDotProduct<numInputs, 1, 1, numInputs>( sampleColumnVector, sampleColumnVector.transpose() );
        Matrix<1, numInputs> output = matrixDotProduct<1, numInputs, numInputs, numInputs>( gradientRowVector, jacobian );

        for ( unsigned int input = 0; input < numInputs; input++ )
        {
            m_InputsGradient.at( batch, input ) = output.at( 0, input );
        }
    }
}

#endif // ACTIVATIONSOFTMAX_HPP