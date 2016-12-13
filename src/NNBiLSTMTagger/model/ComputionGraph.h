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
	WindowBuilder _word_window;
	vector<UniNode> _hidden_layer;
	LSTMBuilder _lstm_left;
	LSTMBuilder _lstm_right;
	vector<BiNode> _lstm_combine;
	vector<LinearNode> _output;
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
		_word_window.resize(sent_size);
		_hidden_layer.resize(sent_size);
		_lstm_left.resize(sent_size);
		_lstm_right.resize(sent_size);
		_lstm_combine.resize(sent_size);
		_output.resize(sent_size);
	}

	inline void clear(){
		_word_inputs.clear();
		_word_window.clear();
		_hidden_layer.clear();
		_lstm_left.clear();
		_lstm_right.clear();
		_lstm_combine.clear();
		_output.clear();
	}

public:
	inline void initial(ModelParams& model_params, HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		int word_max_size = _word_inputs.size();
		for (int idx = 0; idx < word_max_size; idx++) {
			_word_inputs[idx].setParam(&model_params.words);
			_word_inputs[idx].init(hyper_params.wordDim, hyper_params.dropProb, mem);
			_hidden_layer[idx].setParam(&model_params._uni_tanh_project);
			_hidden_layer[idx].init(hyper_params.hiddenSize, hyper_params.dropProb, mem);
			_lstm_combine[idx].setParam(&model_params._bi_tanh_project);
			_lstm_combine[idx].init(hyper_params.rnnHiddenSize, -1, mem);
			_output[idx].setParam(&model_params._linear_project);
			_output[idx].init(hyper_params.labelSize, -1, mem);
		}
		_word_window.init(hyper_params.wordDim, hyper_params.wordContext, mem);
		_lstm_left.init(&model_params._lstm_left_project, hyper_params.dropProb, true, mem);
		_lstm_right.init(&model_params._lstm_right_project, hyper_params.dropProb, false, mem);
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
		_word_window.forward(this, getPNodes(_word_inputs, seq_size));
		for (int idx = 0; idx < seq_size; idx++) {
			_hidden_layer[idx].forward(this, &_word_window._outputs[idx]);
		}
		_lstm_left.forward(this, getPNodes(_hidden_layer, seq_size));
		_lstm_right.forward(this, getPNodes(_hidden_layer, seq_size));
		for (int idx = 0; idx < seq_size; idx++) {
			_lstm_combine[idx].forward(this, &_lstm_left._hiddens[idx], &_lstm_right._hiddens[idx]);
			_output[idx].forward(this, &_lstm_combine[idx]);
		}
	}

};

#endif/*SRC_ComputionGraph_H_*/