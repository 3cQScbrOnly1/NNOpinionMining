#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	LookupTable words;
	RNNParams _rnn_layer;

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
		_rnn_layer.initial(hyper_params.labelSize, hyper_params.wordDim, mem);
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&words.E, "words.E");
	}

	void saveModel(){
	}
	void loadModel(const string& infile){
	}
};

#endif  /*SRC_ModelParams_H_*/