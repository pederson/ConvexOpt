#include <iostream>

#include "include/Gradient.hpp"

using namespace std;

// compile me with g++ -std=c++11 cvx_test.cpp -o cvx_test

int main(int argc, char * argv[]){

	unsigned int niters = 20;
	Matrix Q = randmat(5,5);
	Vector b = randvec(5);
	oracle_fn oracle = [Q,b](const Vector & x)->Vector{return (~Q)*Q*x+b;};
	cvx_fn func = [Q,b](const Vector & x)->double{Vector Qx = Q*x; return 0.5*Vector::dot(Qx,Qx)+Vector::dot(b,x);};
	
	// constant step function 
	stepsize_fn const_step = [](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return constant_step(0.05);};

	// decreasing step function
	stepsize_fn decr_step = [](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return inverse_step(2,2,k);};

	// backtracking line search step
	stepsize_fn btls = [func](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return btls_step(0.4, 0.9, x, d, gf, func);};

	// try with a constant step 
	cout << "************* GD - CONSTANT STEP ************" << endl;
	Vector x = randvec(5);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::gd_step(x, oracle, const_step, k);
		// cout << "x(" << k+1 << "): " << x ; 
		cout << " f[x(" << k+1 << ")]: " << func(x) << endl;
	}


	cout << "************* GD - DECREASING STEP ************" << endl;
	// try with a decreasing step size
	x = randvec(5);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::gd_step(x, oracle, decr_step, k);
		// cout << "x(" << k+1 << "): " << x ; 
		cout << " f[x(" << k+1 << ")]: " << func(x) << endl;
	}

	cout << "************* GD - BTLS STEP ************" << endl;
	// try with a decreasing step size
	x = randvec(5);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::gd_step(x, oracle, btls, k);
		// cout << "x(" << k+1 << "): " << x ; 
		cout << " f[x(" << k+1 << ")]: " << func(x) << endl;
	}
	
	// cout << "x(1): " << xnew << " f(x(1)): " << func(xnew) << endl;
	return 0;
}