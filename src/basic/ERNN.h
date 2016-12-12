#ifndef ERNN
#define ERNN

#include "BiOP.h"
#include "Graph.h"

struct ERNNParams {
	BiParams _rnn;

	ERNNParams() {
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		_rnn.exportAdaParams(ada);
	}

	inline void initial(int nOSize, int nISize, AlignedMemoryPool* mem = NULL) {
		_rnn.initial(nOSize, nOSize, nISize, true, mem);
	}

	inline int inDim(){
			return _rnn.W2.inDim();
	}

	inline int outDim(){
			return _rnn.W2.outDim();
	}

};

class ERNNBuilder{
public:
	int _nSize;
	int _inDim;
	int _outDim;

	Node _bucket;
	vector<BiNode> _output;
	
	ERNNParams* _params;
	
	bool _left2right;

public:
	~ERNNBuilder() {
		clear();
	}

	ERNNBuilder() {
		clear();
	}

public:
	inline void init(ERNNParams* paramInit, dtype dropout, bool left2right = true, AlignedMemoryPool* mem = NULL) {
		_params = paramInit;
		_inDim = _params->_rnn.W2.inDim();
		_outDim = _params->_rnn.W2.outDim();
		int maxsize = _output.size();
		for (int idx = 0; idx < maxsize; idx++){
			_output[idx].setParam(&_params->_rnn);
			_output[idx].setFunctions(&fsigmoid, &dsigmoid);
			_output[idx].init(_outDim, dropout, mem);
		}
		_left2right = left2right;
		_bucket.init(_outDim, -1, mem);
	}
	
	inline void resize(int maxsize){
		_output.resize(maxsize);
	}	
	
	inline void clear() {
		_output.clear();
		
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
		_left2right = true;	
		_params = NULL;
	}	
	
	inline void forward(Graph *cg, const vector<PNode>& x) {
		if (x.size() == 0){
			std::cout << "empty inputs for ERNN operation" << std::endl;
			return;
		}
		_nSize = x.size();
		if (x[0]->val.dim != _inDim) {
			std::cout << "input dim dose not match for seg operation" << std::endl;
			return;
		}
		if (_left2right)
			left2right_forward(cg, x);
		else
			right2left_forward(cg, x);
	}
	
protected:
	inline void left2right_forward(Graph *cg, const vector<PNode>& x) {
		for (int idx = 0; idx < _nSize; idx++) {
			if (idx == 0)
				_output[idx].forward(cg, &_bucket, x[idx]);
			else
				_output[idx].forward(cg, &_output[idx - 1], x[idx]);
		}
	}
	inline void right2left_forward(Graph *cg, const vector<PNode>& x) {
		for (int idx = _nSize - 1; idx >= 0; idx--) {
			if (idx == _nSize - 1)
				_output[idx].forward(cg, &_bucket, x[idx]);
			else
				_output[idx].forward(cg, &_output[idx + 1], x[idx]);
		}
	}
};

class IncERNNBuilder{
public:
	int _nSize;
	int _inDim;
	int _outDim;

	Node _bucket;
	BiNode _output;
	
	ERNNParams* _params;

public:
	~IncERNNBuilder() {
		clear();
	}

	IncERNNBuilder() {
		clear();
	}

public:
	inline void init(ERNNParams* paramInit, dtype dropout, AlignedMemoryPool* mem = NULL) {
		_params = paramInit;
		_inDim = _params->_rnn.W2.inDim();
		_outDim = _params->_rnn.W2.outDim();

		_output.setParam(&_params->_rnn);
		_output.setFunctions(&ftanh, &dtanh);
		_output.init(_outDim, dropout, mem);

		_bucket.init(_outDim, -1, mem);
	}
	
	inline void clear() {	
		_nSize = 0;
		_inDim = 0;
		_outDim = 0;
		_params = NULL;
	}	
	
public:
	inline void forward(Graph *cg, PNode x, IncERNNBuilder* prev = NULL) {
		if (prev == NULL)
			_output.forward(cg, &_bucket, x);
		else
			_output.forward(cg, &prev->_output, x);
	}
};
#endif