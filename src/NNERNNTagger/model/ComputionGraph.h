#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"

struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;
	const static int max_char_length = 32;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	RNNBuilder _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_size, int char_size){
		_word_inputs.resize(sent_size);
		_output.resize(sent_size);
	}

	inline void clear(){
		_word_inputs.clear();
		_output.clear();
	}

public:
	inline void initial(ModelParams& model_params, HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		int word_max_size = _word_inputs.size();
		for (int idx = 0; idx < word_max_size; idx++) {
			_word_inputs[idx].setParam(&model_params.words);
			_word_inputs[idx].init(hyper_params.wordDim, hyper_params.dropProb, mem);
		}
		_output.init(&model_params._rnn_layer, hyper_params.dropProb, true, mem);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		int seq_size;
		max_sentence_length > features.size() ? seq_size = features.size() : seq_size = max_sentence_length;
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			_word_inputs[idx].forward(this, feature.words[0]);
		}
		_output.forward(this, getPNodes(_word_inputs, seq_size));
	}

};

#endif/*SRC_ComputionGraph_H_*/