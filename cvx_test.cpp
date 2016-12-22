#include <iostream>

#include "include/Gradient.hpp"

using namespace std;

// compile me with g++ -std=c++11 cvx_test.cpp -o cvx_test

int main(int argc, char * argv[]){

	unsigned int niters = 20;
	unsigned int vecsize = 10;
	Matrix Q = randmat(vecsize, vecsize);
	Matrix A = (~Q)*Q;
	Vector b = randvec(vecsize);
	oracle_fn oracle = [Q,b](const Vector & x)->Vector{return (~Q)*(Q*x-b);};
	cvx_fn func = [Q,b](const Vector & x)->double{return 0.5*norm_2(Q*x-b);};
	
	double mu=1e-2;
	cvx_fn func_lasso = [Q,b,mu](const Vector & x)->double{return 0.5*norm_2(Q*x-b) + mu*norm_1(x);};
	oracle_fn subgrad = [Q,b,mu](const Vector & x)->Vector{return (~Q)*(Q*x-b)+mu*sign(x);};
	proximal_fn prox_lasso = [](const Vector & x)->Vector{
														  Vector out(x);
														  for (auto i=0; i<x.length(); i++){
														  	if (x(i) >= 1) out(i) = x(i)-1;
														  	else if (x(i) <= 1) out(i) = x(i)+1;
														  	else out(i) = 0;
														  }
														  return out;
	};

	// constant step function 
	stepsize_fn const_step = [](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return constant_step(0.01);};

	// decreasing step function
	stepsize_fn decr_step = [](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return inverse_step(1,2,k);};

	// backtracking line search step
	stepsize_fn btls = [func](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return btls_step(0.4, 0.9, x, d, gf, func);};

	// try with a constant step 
	cout << "************* GD - CONSTANT STEP ************" << endl;
	Vector x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::gd_step(x, oracle, const_step, k);
		// cout << "x(" << k+1 << "): " << x ; 
		cout << " f[x(" << k+1 << ")]: " << func(x) << endl;
	}


	cout << "************* GD - DECREASING STEP ************" << endl;
	// try with a decreasing step size
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::gd_step(x, oracle, decr_step, k);
		// cout << "x(" << k+1 << "): " << x ; 
		cout << " f[x(" << k+1 << ")]: " << func(x) << endl;
	}

	cout << "************* GD - BTLS STEP ************" << endl;
	// try with a decreasing step size
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::gd_step(x, oracle, btls, k);
		// cout << "x(" << k+1 << "): " << x ; 
		cout << " f[x(" << k+1 << ")]: " << func(x) << endl;
	}


	cout << "************* SUBGD - BTLS STEP ************" << endl;
	// try with a decreasing step size
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::subgd_step(x, subgrad, btls, k);
		// cout << "x(" << k+1 << "): " << x ; 
		cout << " f[x(" << k+1 << ")]: " << func_lasso(x) << endl;
	}


	cout << "************* PROXGD - BTLS STEP ************" << endl;
	// try with a decreasing step size
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::proxgd_step(x, subgrad, btls, prox_lasso, k);
		// cout << "x(" << k+1 << "): " << x ; 
		cout << " f[x(" << k+1 << ")]: " << func_lasso(x) << endl;
	}


	partial_grad_fn partial_oracle = [Q,b](const Vector & x, unsigned int i)->Vector{return ~(2*(Vector::dot(Q.row(i),x)-b(i))*(Q.row(i)));};
	cout << "************* SGD - CONSTANT STEP ************" << endl;
	// try with a decreasing step size
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::sgd_step(x, partial_oracle, const_step, k, A.rows());
		cout << " f[x(" << k+1 << ")]: " << func(x) << endl;
	}



	cout << "************* SVRG - CONSTANT STEP ************" << endl;
	// try with a decreasing step size
	x = randvec(vecsize);
	Vector x_old = x;
	Vector full_grad = oracle(x_old);
	for (unsigned int m=0; m<5; m++){
		x_old = x;
		full_grad = oracle(x_old);
		for (unsigned int k=0; k<niters; k++){
			x = gradient::svrg_step(x, x_old, partial_oracle, full_grad, const_step, k, A.rows());
			cout << " f[x(" << m*niters+k+1 << ")]: " << func(x) << endl;
		}
	}
	

	return 0;
}