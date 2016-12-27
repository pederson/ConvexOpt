#include <iostream>

#include "include/Gradient.hpp"

using namespace std;

// compile me with g++ -std=c++11 cvx_test.cpp -o cvx_test

int main(int argc, char * argv[]){

	// define the data sets
	unsigned int niters = 20;
	unsigned int vecsize = 10;
	Matrix Q = randmatn(10*vecsize, vecsize);
	Vector b = randvecn(10*vecsize);
	SparseMatrix S = sprandmatn(100*vecsize, vecsize, 0.3);
	cout << S.nnz() << "/" << S.rows()*S.cols() << endl;


	// ********** DEFINE THE MINIMIZATION PROBLEMS **************
	// Function 1 ---> min 1/2 ||Q*x-b||^2
	cvx_fn 				func1 = [Q,b](const Vector & x)->double{return 0.5*norm_2(Q*x-b);};
	oracle_fn 			grad1 = [Q,b](const Vector & x)->Vector{return (~Q)*(Q*x-b);};
	partial_grad_fn 	partial1 = [Q,b](const Vector & x, unsigned int i)->Vector{return ~((Vector::dot(Q.row(i),x)-b(i))*(Q.row(i)));};
	
	// Function 2 (LASSO) ---> min 1/2 ||Q*x-b||^2 + mu*||x||_1
	double mu=1e-6;
	double gam=2;
	cvx_fn 			func2 = [Q,b,mu](const Vector & x)->double{return 0.5*norm_2(Q*x-b) + mu*norm_1(x);};
	oracle_fn 		subgrad2 = [Q,b,mu](const Vector & x)->Vector{return (~Q)*(Q*x-b)+mu*sign(x);};
	proximal_fn 	prox2 = [mu](const Vector & x)->Vector{return sign(x).elem_mult((abs(x)-mu));};
	oracle_fn 		fws2 = [grad1,gam](const Vector & x)->Vector{Vector gd = grad1(x);
																 Vector out(x.length()); out.fill(0);
																 out(argmin(gd)) = -gam*sgn(gd(argmin(gd)));
																 return out;};

	// Function 3 (Sparse Regression) ---> min 1/2 ||Sx-b||^2
	// cvx_fn 		func3 = [S,b](const SparseVector & x)->double{return 0.5*norm_2(S*x-b);};

	// ***********************************************************


	// ************** DEFINE STEP SIZE FUNCTIONS *****************
	// constant step function 
	stepsize_fn const_step = [](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return constant_step(0.01);};
	// decreasing step function
	stepsize_fn decr_step = [](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return inverse_step(1,50,k);};
	// backtracking line search step
	stepsize_fn btls1 = [func1](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return btls_step(0.4, 0.9, x, d, gf, func1);};
	stepsize_fn btls2 = [func2](unsigned int k, const Vector & x, const Vector & d, const Vector & gf)->double{return btls_step(0.4, 0.9, x, d, gf, func2);};
	// ***********************************************************




// **************** PROBLEM 1 ********************
	// try with a constant step 
	cout << "************* GD - CONSTANT STEP ************" << endl;
	Vector x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::gd_step(x, grad1, const_step, k);
		cout << " f[x(" << k+1 << ")]: " << func1(x) << endl;
	}


	cout << "************* GD - DECREASING STEP ************" << endl;
	// try with a decreasing step size
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::gd_step(x, grad1, decr_step, k);
		cout << " f[x(" << k+1 << ")]: " << func1(x) << endl;
	}

	cout << "************* GD - BTLS STEP ************" << endl;
	// try with a btls step size
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::gd_step(x, grad1, btls1, k);
		cout << " f[x(" << k+1 << ")]: " << func1(x) << endl;
	}

	cout << "************* SGD - CONSTANT STEP ************" << endl;
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::sgd_step(x, partial1, const_step, k, Q.rows());
		cout << " f[x(" << k+1 << ")]: " << func1(x) << endl;
	}



	cout << "************* SVRG - CONSTANT STEP ************" << endl;
	x = randvec(vecsize);
	x = gradient::gd_step(x, grad1, const_step, 0);
	Vector x_old = x;
	Vector full_grad = grad1(x_old)/Q.rows();
	for (unsigned int m=0; m<5; m++){
		
		for (unsigned int k=0; k<niters; k++){
			x = gradient::svrg_step(x, x_old, partial1, full_grad, const_step, m*niters+k, Q.rows());
			cout << " f[x(" << m*niters+k+1 << ")]: " << func1(x) << endl;
		}
		x_old = x;
		full_grad = grad1(x_old)/Q.rows();
	}



	




// **************** PROBLEM 2 ********************
	cout << "************* SUBGD - CONSTANT STEP ************" << endl;
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::subgd_step(x, subgrad2, btls2, k);
		cout << " f[x(" << k+1 << ")]: " << func2(x) << endl;
	}


	cout << "************* PROXGD - CONSTANT STEP ************" << endl;
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::proxgd_step(x, grad1, const_step, prox2, k);
		cout << " f[x(" << k+1 << ")]: " << func2(x) << endl;
	}


	cout << "************* FRANK-WOLFE - DECREASING STEP ************" << endl;
	x = randvec(vecsize);
	for (unsigned int k=0; k<niters; k++){
		x = gradient::fw_step(x, fws2, decr_step, k);
		cout << " f[x(" << k+1 << ")]: " << func2(x) << endl;
	}


	cout << "************* FISTA - DECREASING STEP ************" << endl;
	x = randvec(vecsize);
	Vector y(x);
	double lambda=0;
	for (unsigned int k=0; k<niters; k++){
		x = gradient::fista_step(x, y, lambda, grad1, const_step, prox2, k);
		cout << " f[x(" << k+1 << ")]: " << func2(x) << endl;
	}



// *************** PROBLEM 3 ********************
	
	

	return 0;
}