#ifndef LAYERDENSE_HPP
#define LAYERDENSE_HPP

/**************************************************************************
 * The LayerDense class implements a neural network single layer.
**************************************************************************/

#define _USE_MATH_DEFINES

#include "Matrix.hpp"

#include <random>

template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
struct LayerDensePartials
{
};

template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
class LayerDense
{
	public:
		LayerDense();
		LayerDense(const Matrix<numInputs, numNeurons>& weights, const Matrix<numBatches, numNeurons>& biases);

		Matrix<numBatches, numNeurons> forwardPass (const Matrix<numBatches, numInputs>& in) const;

		void backwardPass (const Matrix<numBatches, numInputs>& in, const Matrix<numBatches, numNeurons>& gradient);

		void updateLayer (float learningRate);

		Matrix<numBatches, numInputs> getInputsGradient() const { return m_InputsGradient; }
		Matrix<numInputs, numNeurons> getWeightsGradient() const { return m_WeightsGradient; }
		Matrix<numBatches, numNeurons> getBiasesGradient() const { return m_BiasesGradient; }

		Matrix<numInputs, numNeurons> getWeights() const { return m_Weights; }
		Matrix<numBatches, numNeurons> getBiases() const { return m_Biases; }

    private:
		Matrix<numInputs, numNeurons> 		m_Weights;
		Matrix<numBatches, numNeurons> 		m_Biases;

		Matrix<numBatches, numInputs> 		m_InputsGradient;
		Matrix<numInputs, numNeurons> 		m_WeightsGradient;
		Matrix<numBatches, numNeurons> 		m_BiasesGradient;
};

template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
LayerDense<numBatches, numInputs, numNeurons>::LayerDense() :
	m_Weights(),
	m_Biases(),
	m_InputsGradient(),
	m_WeightsGradient(),
	m_BiasesGradient()
{
	// generate random values for weights
	std::random_device rd;
	std::mt19937 gen( rd() );
	constexpr int maxVal = 100000;
	// TODO should this be gaussian distribuition?
	std::uniform_int_distribution<> distr( 0, maxVal );
	constexpr float oneOverMaxVal = 1.0f / static_cast<float>( maxVal );
	constexpr float minimizer = 0.3f;

	for ( unsigned int row = 0; row < numInputs; row++ )
	{
		for ( unsigned int col = 0; col < numNeurons; col++ )
		{
			float randVal = ( (static_cast<float>(distr(gen)) * oneOverMaxVal * 2.0f) - 1.0f ) * minimizer;
			m_Weights.at( row, col ) = randVal;
		}
	}
}

template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
LayerDense<numBatches, numInputs, numNeurons>::LayerDense (const Matrix<numInputs, numNeurons>& weights, const Matrix<numBatches, numNeurons>& biases) :
	m_Weights( weights ),
	m_Biases( biases ),
	m_InputsGradient(),
	m_WeightsGradient(),
	m_BiasesGradient()
{
}

template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
Matrix<numBatches, numNeurons> LayerDense<numBatches, numInputs, numNeurons>::forwardPass (const Matrix<numBatches, numInputs>& in) const
{
	Matrix<numBatches, numNeurons> matOut = matrixDotProduct( in, m_Weights ) + m_Biases;

	return matOut;
}


template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
void LayerDense<numBatches, numInputs, numNeurons>::backwardPass (const Matrix<numBatches, numInputs>& in, const Matrix<numBatches, numNeurons>& gradient)
{
	m_InputsGradient = matrixDotProduct<numBatches, numNeurons, numNeurons, numInputs>( gradient, m_Weights.transpose() );
	m_WeightsGradient = matrixDotProduct<numInputs, numBatches, numBatches, numNeurons>( in.transpose(), gradient ); 
	Matrix<numBatches, numNeurons> biasesGradient;
    for ( unsigned int batch = 0; batch < numBatches; batch++ )
    {
        float sum = 0.0f;
        for ( unsigned int inputNum = 0; inputNum < numNeurons; inputNum++ )
        {
            sum += gradient.at( inputNum, batch );
        }
        for ( unsigned int inputNum = 0; inputNum < numNeurons; inputNum++ )
        {
            biasesGradient.at( batch, inputNum ) = sum;
        }
    }
	m_BiasesGradient = biasesGradient;
}

template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
void LayerDense<numBatches, numInputs, numNeurons>::updateLayer (float learningRate)
{
	m_Weights += m_WeightsGradient * ( -1.0f * learningRate );
	m_Biases += m_BiasesGradient * ( -1.0f * learningRate );
}

#endif // LAYERDENSE_HPP