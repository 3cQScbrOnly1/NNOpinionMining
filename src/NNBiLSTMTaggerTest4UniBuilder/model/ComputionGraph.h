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
	vector<LookupNode> _ext_word_inputs;
	vector<ConcatNode> _word_concats;
	WindowBuilder _word_window;
	//vector<UniNode> _hidden_layer;
	UniBuilder _hidden_layer;
	LSTMBuilder _lstm_left;
	LSTMBuilder _lstm_right;
	vector<BiNode> _lstm_combine;
	vector<LinearNode> _output;

	unordered_map<string, int>* p_word_stats;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_size){
		_word_inputs.resize(sent_size);
		_ext_word_inputs.resize(sent_size);
		_word_concats.resize(sent_size);
		_word_window.resize(sent_size);
		_hidden_layer.resize(sent_size);
		_lstm_left.resize(sent_size);
		_lstm_right.resize(sent_size);
		_lstm_combine.resize(sent_size);
		_output.resize(sent_size);
	}

	inline void clear(){
		_word_inputs.clear();
		_ext_word_inputs.clear();
		_word_concats.clear();
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
			_word_inputs[idx].setParam(&model_params._words);
			_word_inputs[idx].init(hyper_params.wordDim, hyper_params.dropProb, mem);
			_ext_word_inputs[idx].setParam(&model_params._ext_words);
			_ext_word_inputs[idx].init(hyper_params.extWordDim, hyper_params.dropProb, mem);
			_word_concats[idx].init(hyper_params.wordDim + hyper_params.extWordDim, -1, mem);
			_lstm_combine[idx].setParam(&model_params._bi_tanh_project);
			_lstm_combine[idx].init(hyper_params.hiddenSize, -1, mem);
			_output[idx].setParam(&model_params._linear_project);
			_output[idx].init(hyper_params.labelSize, -1, mem);
		}
		_hidden_layer.init(&model_params._uni_tanh_project, hyper_params.dropProb, mem);
		_word_window.init(hyper_params.wordDim + hyper_params.extWordDim, hyper_params.wordContext, mem);
		_lstm_left.init(&model_params._lstm_left_project, hyper_params.dropProb, true, mem);
		_lstm_right.init(&model_params._lstm_right_project, hyper_params.dropProb, false, mem);
		p_word_stats = hyper_params.hyper_word_stats;
	}

public:
	string p_change_word(const string& word){
		double p = 0.5;
		unordered_map<string, int>::iterator it;
		it = p_word_stats->find(word);
		if (it != p_word_stats->end() && it->second == 1)
		{
			double x = rand() / double(RAND_MAX);
			if (x > p)
				return unknownkey;
			else
				return word;
		}
		else
			return word;
	}

public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		int seq_size;
		max_sentence_length > features.size() ? seq_size = features.size() : seq_size = max_sentence_length;
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			string the_word;
			if (bTrain)
				the_word = p_change_word(feature.words[0]);
			else
				the_word = feature.words[0];

			_word_inputs[idx].forward(this, the_word);
			_ext_word_inputs[idx].forward(this, the_word);
		}
		for (int idx = 0; idx < seq_size; idx++) {
			_word_concats[idx].forward(this, &_word_inputs[idx], &_ext_word_inputs[idx]);
		}
		_word_window.forward(this, getPNodes(_word_concats , seq_size));
		_hidden_layer.forward(this, getPNodes(_word_window._outputs, seq_size));
		_lstm_left.forward(this, getPNodes(_hidden_layer._outputs, seq_size));
		_lstm_right.forward(this, getPNodes(_hidden_layer._outputs, seq_size));
		for (int idx = 0; idx < seq_size; idx++) {
			_lstm_combine[idx].forward(this, &_lstm_left._hiddens[idx], &_lstm_right._hiddens[idx]);
			_output[idx].forward(this, &_lstm_combine[idx]);
		}
	}

};

#endif/*SRC_ComputionGraph_H_*/