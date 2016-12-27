#ifndef _NEWTON_H
#define _NEWTON_H

#include <functional>
#include <random>
#include <ctime>

#include "ConvexCommon.hpp"

#include "../LinearAlgebra/Matrix.hpp"
#include "../LinearAlgebra/SparseVector.hpp"
#include "../LinearAlgebra/LinearSolvers.hpp"


namespace newton{

	// one step of a generic stepping algorithm
	// given the previous step x,
	// an oracle function that returns the new direction at a point x
	// a step size function that returns the step size
	Vector generic_step(const Vector & x, 
				   oracle_fn direction,
				   stepsize_fn stepsize,
				   unsigned int k)
	{
		// get gradient direction
		Vector d = oracle(x);
		// get stepsize
		double gamma = stepsize(k, x, d, d);
		// take step
		Vector xnew = x + gamma*d;
		return xnew;
	}



	// one step of a full newton algorithm
	// given the previous step x,
	// an oracle function that returns the new direction at a point x
	// a step size function that returns the step size
	Vector newton_step(const Vector & x, 
				   oracle_fn oracle,
				   hessian_fn hessian,
				   stepsize_fn stepsize,
				   unsigned int k)
	{
		// get gradient direction
		Vector grad = oracle(x);

		// get hessian
		Matrix H = hessian(x);

		// get search direction
		Matrix Q, R, U;
		qr_householder(H, U, R, Q);
		Vector hold = unitary_solve(Q, grad);
		Vector d = upper_triangular_solve(R, hold);

		// get stepsize
		double gamma = stepsize(k, x, p, grad);

		// take step
		Vector xnew = x + gamma*p;
		return xnew;
	}


	// one step of the newton-like DFP algorithm
	// given the previous step x,
	// an oracle function that returns the new direction at a point x
	// a step size function that returns the step size
	Vector dfp_step(const Vector & x, 
				   Matrix & B,
				   Vector & old_grad,
				   oracle_fn oracle,
				   stepsize_fn stepsize,
				   unsigned int k)
	{
		// get gradient direction
		Vector grad = oracle(x);
		Vector y = grad - old_grad;

		// get search direction
		Matrix Q, R, U;
		qr_householder(B, U, R, Q);
		Vector hold = unitary_solve(Q, y);
		Vector s = upper_triangular_solve(R, hold);

		// take step
		Vector xnew = x + s;

		// update B
		double rho = Vector::dot(y,s);
		Matrix Z = eye(B.rows()) - rho*s*(~y);
		B = (~Z)*B*Z + rho*y*(~y);

		return xnew;
	}


	// one step of the newton-like BFGS algorithm
	// given the previous step x,
	// an oracle function that returns the new direction at a point x
	// a step size function that returns the step size
	Vector bfgs_step(const Vector & x, 
				   Matrix & H,
				   Vector & old_grad,
				   oracle_fn oracle,
				   stepsize_fn stepsize,
				   unsigned int k)
	{
		// get gradient direction
		Vector grad = oracle(x);
		Vector y = grad - old_grad;

		// get search direction
		Vector s = H*y;

		// take step
		Vector xnew = x + s;

		// update H
		double rho = Vector::dot(y,s);
		Matrix Z = eye(B.rows()) - rho*s*(~y);
		H = (~Z)*H*Z + rho*s*(~s);

		return xnew;
	}
}


#endif