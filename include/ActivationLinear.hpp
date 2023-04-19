#ifndef ACTIVATIONLINEAR_HPP
#define ACTIVATIONLINEAR_HPP

/**************************************************************************
 * The ActivationLinear class implements the linear activation function.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

#include <algorithm>

template <unsigned int numBatches, unsigned int numInputs>
class ActivationLinear
{
	public:
		ActivationLinear();

		Matrix<numBatches, numInputs> forwardPass (const Matrix<numBatches, numInputs>& in) const;

        void backwardPass (const Matrix<numBatches, numInputs>& in, const Matrix<numBatches, numInputs>& gradient);

        Matrix<numBatches, numInputs> getInputsGradient() const { return m_Gradient; }

    private:
        Matrix<numBatches, numInputs>   m_Gradient;
};

template <unsigned int numBatches, unsigned int numInputs>
ActivationLinear<numBatches, numInputs>::ActivationLinear()
{
}

template <unsigned int numBatches, unsigned int numInputs>
Matrix<numBatches, numInputs> ActivationLinear<numBatches, numInputs>::forwardPass (const Matrix<numBatches, numInputs>& in) const
{
	return in;
}

template <unsigned int numBatches, unsigned int numInputs>
void ActivationLinear<numBatches, numInputs>::backwardPass (const Matrix<numBatches, numInputs>& in, const Matrix<numBatches, numInputs>& gradient)
{
    m_Gradient = gradient;
}

#endif // ACTIVATIONLINEAR_HPP