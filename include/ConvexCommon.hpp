#ifndef _CONVEXCOMMON_H
#define _CONVEXCOMMON_H

#include <functional>

#include "../LinearAlgebra/Matrix.hpp"
#include "../LinearAlgebra/SparseVector.hpp"
#include "../LinearAlgebra/LinearSolvers.hpp"


typedef std::function<double(const Vector &)> cvx_fn;
typedef std::function<Vector(const Vector &)> oracle_fn;
typedef std::function<Vector(const Vector &, unsigned int)> partial_grad_fn;
typedef std::function<double(unsigned int, const Vector &, const Vector &, const Vector &)> stepsize_fn;
typedef std::function<Vector(const Vector &)> projection_fn;
typedef std::function<Vector(const Vector &)> proximal_fn;

typedef std::function<double(const SparseVector &)> sp_cvx_fn;
typedef std::function<SparseVector(const SparseVector &)> sp_oracle_fn;
typedef std::function<SparseVector(const SparseVector &, unsigned int)> sp_partial_grad_fn;
typedef std::function<double(unsigned int, const SparseVector &, const SparseVector &, const SparseVector &)> sp_stepsize_fn;


// constant step size
inline double constant_step(double a) {return a;};

// step size schedule of the form a/(k+b), where k is the current iteration
inline double inverse_step(double a, double b, unsigned int k){return a/(k+b);};

// backtracking line search step
inline double btls_step(double a, double b, 
						const Vector & x, const Vector & d,
						const Vector & grad_f,
						cvx_fn f)
{
	double t=1.0;
	while (f(x+t*d) > (f(x)+a*t*Vector::dot(grad_f,d))){
		t *= b;
	}
	return t;
}

// sparse backtracking line search step
inline double sp_btls_step(double a, double b, 
						const SparseVector & x, const SparseVector & d,
						const SparseVector & grad_f,
						sp_cvx_fn f)
{
	double t=1.0;
	while (f(x+t*d) > (f(x)+a*t*SparseVector::dot(grad_f,d))){
		t *= b;
	}
	return t;
}

#endif