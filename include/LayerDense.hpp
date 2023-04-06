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
		LayerDense(float weightL1RegularizationStrength = 0.0f, float biasL1RegularizationStrength = 0.0f,
					float weightL2RegularizationStrength = 0.0f, float biasL2RegularizationStrength = 0.0f);
		LayerDense(const Matrix<numInputs, numNeurons>& weights, const Matrix<numBatches, numNeurons>& biases, float weightL1RegularizationStrength = 0.0f,
					float biasL1RegularizationStrength = 0.0f, float weightL2RegularizationStrength = 0.0f, float biasL2RegularizationStrength = 0.0f);

		Matrix<numBatches, numNeurons> forwardPass (const Matrix<numBatches, numInputs>& in) const;

		void backwardPass (const Matrix<numBatches, numInputs>& in, const Matrix<numBatches, numNeurons>& gradient);

		void updateLayer (float learningRate, float momemtum = 0.0f);

		Matrix<numBatches, numInputs> getInputsGradient() const { return m_InputsGradient; }
		Matrix<numInputs, numNeurons> getWeightsGradient() const { return m_WeightsGradient; }
		Matrix<numBatches, numNeurons> getBiasesGradient() const { return m_BiasesGradient; }

		Matrix<numInputs, numNeurons> getWeights() const { return m_Weights; }
		Matrix<numBatches, numNeurons> getBiases() const { return m_Biases; }

		float getRegularizationLoss() const;

    private:
		Matrix<numInputs, numNeurons> 		m_Weights;
		Matrix<numBatches, numNeurons> 		m_Biases;

		Matrix<numBatches, numInputs> 		m_InputsGradient;
		Matrix<numInputs, numNeurons> 		m_WeightsGradient;
		Matrix<numBatches, numNeurons> 		m_BiasesGradient;

		Matrix<numInputs, numNeurons> 		m_WeightsMomentum;
		Matrix<numBatches, numNeurons> 		m_BiasesMomentum;

		float 								m_WeightL1RegularizationStrength;
		float 								m_BiasL1RegularizationStrength;
		float 								m_WeightL2RegularizationStrength;
		float 								m_BiasL2RegularizationStrength;
};

template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
LayerDense<numBatches, numInputs, numNeurons>::LayerDense (float weightL1RegularizationStrength, float biasL1RegularizationStrength,
					float weightL2RegularizationStrength, float biasL2RegularizationStrength) :
	m_Weights(),
	m_Biases(),
	m_InputsGradient(),
	m_WeightsGradient(),
	m_BiasesGradient(),
	m_WeightsMomentum(),
	m_BiasesMomentum(),
	m_WeightL1RegularizationStrength( weightL1RegularizationStrength ),
	m_BiasL1RegularizationStrength( biasL1RegularizationStrength ),
	m_WeightL2RegularizationStrength( weightL2RegularizationStrength ),
	m_BiasL2RegularizationStrength( biasL2RegularizationStrength )
{
	// generate random values for weights
	std::random_device rd;
	std::mt19937 gen( rd() );
	constexpr int maxVal = 100000;
	std::normal_distribution<> distr( 0.0f, 0.25f );
	constexpr float minimizer = 0.01f;

	for ( unsigned int row = 0; row < numInputs; row++ )
	{
		for ( unsigned int col = 0; col < numNeurons; col++ )
		{
			m_Weights.at( row, col ) = distr( gen ) * minimizer;
		}
	}
}

template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
LayerDense<numBatches, numInputs, numNeurons>::LayerDense (const Matrix<numInputs, numNeurons>& weights, const Matrix<numBatches, numNeurons>& biases,
					float weightL1RegularizationStrength, float biasL1RegularizationStrength, float weightL2RegularizationStrength,
					float biasL2RegularizationStrength) :
	m_Weights( weights ),
	m_Biases( biases ),
	m_InputsGradient(),
	m_WeightsGradient(),
	m_BiasesGradient(),
	m_WeightsMomentum(),
	m_BiasesMomentum(),
	m_WeightL1RegularizationStrength( weightL1RegularizationStrength ),
	m_BiasL1RegularizationStrength( biasL1RegularizationStrength ),
	m_WeightL2RegularizationStrength( weightL2RegularizationStrength ),
	m_BiasL2RegularizationStrength( biasL2RegularizationStrength )
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
	for ( unsigned int inputNum = 0; inputNum < numNeurons; inputNum++ )
	{
		float sum = 0.0f;
		for ( unsigned int batch = 0; batch < numBatches; batch++ )
		{
			sum += gradient.at( batch, inputNum );
		}

		for ( unsigned int batch = 0; batch < numBatches; batch++ )
		{
			biasesGradient.at( batch, inputNum ) = sum;
		}
	}

	m_BiasesGradient = biasesGradient;

	// l1 regularization
	if ( m_WeightL1RegularizationStrength > 0.0f )
	{
		Matrix<numInputs, numNeurons> l1TempWeights;
		for ( unsigned int inputNum = 0; inputNum < numInputs; inputNum++ )
		{
			for ( unsigned int neuronNum = 0; neuronNum < numNeurons; neuronNum++ )
			{
				l1TempWeights.at( inputNum, neuronNum ) = ( m_Weights.at(inputNum, neuronNum) < 0.0f ) ? -1.0f : 1.0f;
			}
		}
		m_WeightsGradient += l1TempWeights * m_WeightL1RegularizationStrength;
	}
	if ( m_BiasL1RegularizationStrength > 0.0f )
	{
		Matrix<numBatches, numNeurons> l1TempBiases;
		for ( unsigned int batch = 0; batch < numBatches; batch++ )
		{
			for ( unsigned int neuronNum = 0; neuronNum < numNeurons; neuronNum++ )
			{
				l1TempBiases.at( batch, neuronNum ) = ( m_Biases.at(batch, neuronNum) < 0.0f ) ? -1.0f : 1.0f;
			}
		}
		m_BiasesGradient += l1TempBiases * m_BiasL1RegularizationStrength;
	}

	// l2 regularization
	if ( m_WeightL2RegularizationStrength > 0.0f )
	{
		m_WeightsGradient += m_Weights * ( 2.0f * m_WeightL2RegularizationStrength );
	}
	if ( m_BiasL2RegularizationStrength > 0.0f )
	{
		m_BiasesGradient += m_Biases * ( 2.0f * m_BiasL2RegularizationStrength );
	}
}

template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
void LayerDense<numBatches, numInputs, numNeurons>::updateLayer (float learningRate, float momentum)
{
	m_WeightsMomentum = ( m_WeightsMomentum * momentum ) - ( m_WeightsGradient * learningRate );
	m_BiasesMomentum = ( m_BiasesMomentum * momentum ) - ( m_BiasesGradient * learningRate );
	m_Weights += m_WeightsMomentum;
	m_Biases += m_BiasesMomentum;
}


template <unsigned int numBatches, unsigned int numInputs, unsigned int numNeurons>
float LayerDense<numBatches, numInputs, numNeurons>::getRegularizationLoss() const
{
	float regularizationLoss = 0.0f;

	// l1 regularization
	if ( m_WeightL1RegularizationStrength > 0.0f )
	{
		float sumOfAbsoluteWeights = 0.0f;
		for ( unsigned int row = 0; row < numInputs; row++ )
		{
			for ( unsigned int col = 0; col < numNeurons; col++ )
			{
				sumOfAbsoluteWeights += std::abs( m_Weights.at(row, col) );
			}
		}
		regularizationLoss += m_WeightL1RegularizationStrength * sumOfAbsoluteWeights;
	}
	if ( m_BiasL1RegularizationStrength > 0.0f )
	{
		float sumOfAbsoluteBiases = 0.0f;
		for ( unsigned int row = 0; row < numBatches; row++ )
		{
			for ( unsigned int col = 0; col < numNeurons; col++ )
			{
				sumOfAbsoluteBiases += std::abs( m_Biases.at(row, col) );
			}
		}
		regularizationLoss += m_BiasL1RegularizationStrength * sumOfAbsoluteBiases;
	}

	// l2 regularization
	if ( m_WeightL2RegularizationStrength > 0.0f )
	{
		float sumOfWeightsSquared = 0.0f;
		for ( unsigned int row = 0; row < numInputs; row++ )
		{
			for ( unsigned int col = 0; col < numNeurons; col++ )
			{
				sumOfWeightsSquared += m_Weights.at( row, col ) * m_Weights.at( row, col );
			}
		}
		regularizationLoss += m_WeightL2RegularizationStrength * sumOfWeightsSquared;
	}
	if ( m_BiasL2RegularizationStrength > 0.0f )
	{
		float sumOfBiasesSquared = 0.0f;
		for ( unsigned int row = 0; row < numBatches; row++ )
		{
			for ( unsigned int col = 0; col < numNeurons; col++ )
			{
				sumOfBiasesSquared += m_Biases.at( row, col ) * m_Biases.at( row, col );
			}
		}
		regularizationLoss += m_BiasL2RegularizationStrength * sumOfBiasesSquared;
	}

	return regularizationLoss;
}

#endif // LAYERDENSE_HPP