#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	LookupTable _words;
	LookupTable _ext_words;
	LSTMParams _lstm_left_project;
	LSTMParams _lstm_right_project;
	UniParams _uni_tanh_project;
	BiParams _bi_tanh_project;
	UniParams _linear_project;

	Alphabet _word_alpha;
	Alphabet _ext_word_alpha;
	Alphabet _label_alpha;
public:
	SoftMaxLoss _loss;
public:
	bool initial(HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		if (_label_alpha.size() <= 0 || _word_alpha.size() <= 0)
			return false;
		hyper_params.labelSize = _label_alpha.size();
		hyper_params.wordDim = _words.nDim;
		hyper_params.extWordDim = _ext_words.nDim;
		hyper_params.wordWindow = hyper_params.wordContext * 2 + 1;
		_uni_tanh_project.initial(hyper_params.hiddenSize, hyper_params.wordWindow * (hyper_params.extWordDim), true, mem);
		_lstm_left_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize, mem);
		_lstm_right_project.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize, mem);
		_bi_tanh_project.initial(hyper_params.hiddenSize, hyper_params.rnnHiddenSize, hyper_params.rnnHiddenSize, true, mem);
		hyper_params.inputSize = hyper_params.hiddenSize;
		_linear_project.initial(hyper_params.labelSize, hyper_params.inputSize, false, mem);
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		_words.exportAdaParams(ada);
		_ext_words.exportAdaParams(ada);
		_lstm_left_project.exportAdaParams(ada);
		_lstm_right_project.exportAdaParams(ada);
		_uni_tanh_project.exportAdaParams(ada);
		_bi_tanh_project.exportAdaParams(ada);
		_linear_project.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&_words.E, "words.E");
		checkgrad.add(&_uni_tanh_project.W, "_uni_tanh_project.W");
		checkgrad.add(&_uni_tanh_project.b, "_uni_tanh_project.b");
		checkgrad.add(&_lstm_left_project.output.W1, "&_lstm_left_project.output.W1");
		checkgrad.add(&_lstm_left_project.output.W2, "&_lstm_left_project.output.W2");
		checkgrad.add(&_bi_tanh_project.W1, "_bi_tanh_project.W1");
		checkgrad.add(&_bi_tanh_project.W2, "_bi_tanh_project.W2");
		checkgrad.add(&_bi_tanh_project.b, "_bi_tanh_project.b");
		checkgrad.add(&_linear_project.W, "_linear_project.W");
	}

	void saveModel(){
	}
	void loadModel(const string& infile){
	}
};

#endif  /*SRC_ModelParams_H_*/