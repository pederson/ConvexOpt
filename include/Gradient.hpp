#include <functional>

#include "../LinearAlgebra/Matrix.hpp"
#include "../LinearAlgebra/SparseVector.hpp"
#include "../LinearAlgebra/LinearSolvers.hpp"


typedef std::function<double(const Vector &)> cvx_fn;
typedef std::function<Vector(const Vector &)> oracle_fn;
typedef std::function<double(unsigned int, const Vector &, const Vector &, const Vector &)> stepsize_fn;
typedef std::function<Vector(const Vector &)> projection_fn;

// constant step size
inline double constant_step(double a) {return a;};

// step size schedule of the form a/(k+b), where k is the current iteration
inline double inverse_step(double a, double b, unsigned int k){return a/(k+b);};

// backtracking line search step
inline double btls_step(double a, double b, 
						const Vector & x, const Vector & d,
						const Vector & grad_f,
						cvx_fn f){
	double t=1.0;
	while (f(x+t*d) > (f(x)+a*t*Vector::dot(grad_f,d))){
		t *= b;
	}
	return t;
}


namespace gradient{

	// one step of gradient descent
	// given the previous step x,
	// an oracle function that returns the gradient at a point x
	// a step size function that returns the step size
	Vector gd_step(const Vector & x, 
				   oracle_fn oracle,
				   stepsize_fn stepsize,
				   unsigned int k){
		// get gradient direction
		Vector grad = oracle(x);
		// get stepsize
		double gamma = stepsize(k, x, -1*grad, grad);
		// take step
		Vector xnew = x - gamma*grad;
		return xnew;
	}


	// one step of subgradient descent
	// given the previous step x,
	// an oracle function that returns a subgradient at a point x
	// a step size function that returns the step size
	Vector subgd_step(const Vector & x, 
				   oracle_fn oracle,
				   stepsize_fn stepsize,
				   unsigned int k){
		// get gradient direction
		Vector subgrad = oracle(x);
		// get stepsize
		double gamma = stepsize(k, x, -1*subgrad, subgrad);
		// take step
		Vector xnew = x - gamma*subgrad;
		return xnew;
	}


}