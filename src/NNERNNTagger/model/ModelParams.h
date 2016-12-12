#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	LookupTable words;
	RNNParams _rnn_layer;
	UniParams _uni_params;

	Alphabet _linear_feature;
	Alphabet _word_alpha;
	Alphabet _char_alpha;
	Alphabet _label_alpha;
public:
	SoftMaxLoss _loss;
public:
	bool initial(HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		if (_label_alpha.size() <= 0 || _word_alpha.size() <= 0)
			return false;
		hyper_params.labelSize = _label_alpha.size();
		hyper_params.linearFeatSize = _linear_feature.size();
		hyper_params.wordDim = words.nDim;
		hyper_params.wordWindow = hyper_params.wordContext * 2 + 1;
		_rnn_layer.initial(hyper_params.rnnHiddenSize, hyper_params.wordDim * hyper_params.wordWindow, mem);
		hyper_params.inputSize = hyper_params.rnnHiddenSize;
		_uni_params.initial(hyper_params.labelSize, hyper_params.inputSize, false, mem);
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		_rnn_layer.exportAdaParams(ada);
		_uni_params.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
		checkgrad.add(&_rnn_layer._rnn.W1, "_rnn_layer._rnn.W1");
		checkgrad.add(&_rnn_layer._rnn.W2, "_rnn_layer._rnn.W2");
		checkgrad.add(&_rnn_layer._rnn.b, "_rnn_layer._rnn.b");
		checkgrad.add(&_uni_params.W, "_uni_params.W");
	}

	void saveModel(){
	}
	void loadModel(const string& infile){
	}
};

#endif  /*SRC_ModelParams_H_*/