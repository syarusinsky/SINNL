#ifndef LAYERDENSE_HPP
#define LAYERDENSE_HPP

/**************************************************************************
 * The LayerDense class implements a neural network single layer.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

#include <random>

template <unsigned int numInputs, unsigned int numNeurons>
class LayerDense
{
	public:
		LayerDense();

		template <unsigned int numBatches>
		Matrix<numBatches, numNeurons> forwardPass (const Matrix<numBatches, numInputs>& in);

    private:
		Matrix<numInputs, numNeurons> 		m_Weights;
		Matrix<numNeurons, numNeurons> 		m_Biases;
};

template <unsigned int numInputs, unsigned int numNeurons>
LayerDense<numInputs, numNeurons>::LayerDense() :
// 	m_Weights({ {{0.2f, 0.5f, -0.26f}, {0.8f, -0.91f, -0.27f}, {-0.5f, 0.26f, 0.17f}, {1.0f, -0.5f, 0.87f}} }),
// 	m_Biases({ {{2.0f, 3.0f, 0.5f}, {2.0f, 3.0f, 0.5f}, {2.0f, 3.0f, 0.5f}} })
	m_Weights(),
	m_Biases()
{
	// generate random values for weights
	std::random_device rd;
	std::mt19937 gen( rd() );
	constexpr int maxVal = 100000;
	std::uniform_int_distribution<> distr( 0, maxVal );
	constexpr float oneOverMaxVal = 1.0f / static_cast<float>( maxVal );
	constexpr float minimizer = 0.3f;

	for ( unsigned int row = 0; row < numInputs; row++ )
	{
		for ( unsigned int col = 0; col < numNeurons; col++ )
		{
			float randVal = static_cast<float>(distr(gen)) * oneOverMaxVal * minimizer;
			m_Weights.at( row, col ) = randVal;
		}
	}
}

template <unsigned int numInputs, unsigned int numNeurons>
template <unsigned int numBatches>
Matrix<numBatches, numNeurons> LayerDense<numInputs, numNeurons>::forwardPass (const Matrix<numBatches, numInputs>& in)
{
	Matrix<numBatches, numNeurons> matOut = matrixDotProduct( in, m_Weights ) + m_Biases;

	return matOut;
}

#endif // LAYERDENSE_HPP