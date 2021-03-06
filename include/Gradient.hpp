#ifndef _GRADIENT_H
#define _GRADIENT_H

#include <functional>
#include <random>
#include <ctime>

#include "ConvexCommon.hpp"

#include "../LinearAlgebra/Matrix.hpp"
#include "../LinearAlgebra/SparseVector.hpp"
#include "../LinearAlgebra/LinearSolvers.hpp"


namespace gradient{

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
		Vector d = direction(x);
		// get stepsize
		double gamma = stepsize(k, x, d, d);
		// take step
		Vector xnew = x + gamma*d;
		return xnew;
	}

	

	// one step of gradient descent
	// given the previous step x,
	// an oracle function that returns the gradient at a point x
	// a step size function that returns the step size
	Vector gd_step(const Vector & x, 
				   oracle_fn oracle,
				   stepsize_fn stepsize,
				   unsigned int k)
	{
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
				   unsigned int k)
	{
		// get gradient direction
		Vector subgrad = oracle(x);
		// get stepsize
		double gamma = stepsize(k, x, -1*subgrad, subgrad);
		// take step
		Vector xnew = x - gamma*subgrad;
		return xnew;
	}



	// one step of proximal gradient descent
	// given the previous step x,
	// an oracle function that returns the gradient at a point x,
	// a step size function that returns the step size,
	// a prox operator that returns the prox value
	Vector proxgd_step(const Vector & x, 
				   oracle_fn oracle,
				   stepsize_fn stepsize,
				   proximal_fn prox,
				   unsigned int k)
	{
		// get gradient direction
		Vector grad = oracle(x);
		// get stepsize
		double gamma = stepsize(k, x, -1*grad, grad);
		// take step
		Vector xnew = x - gamma*grad;
		return prox(xnew);
	}


	// one step of stochastic gradient descent
	// given the previous step x,
	// an oracle function that returns the gradient at a point x,
	// a step size function that returns the step size,
	// a prox operator that returns the prox value
	Vector sgd_step(const Vector & x, 
				   partial_grad_fn partial_grad,
				   stepsize_fn stepsize,
				   unsigned int k,
				   unsigned int n)
	{

		// get a random row
		unsigned int i=rand()%(n-1);

		// get gradient direction
		Vector pgrad = partial_grad(x, i);

		// get stepsize
		double gamma = stepsize(k, x, -1*pgrad, pgrad);

		// take step
		Vector xnew = x - gamma*pgrad;
		return xnew;
	}

	// ******* SPARSE *********
	// one step of stochastic gradient descent
	// given the previous step x,
	// an oracle function that returns the gradient at a point x,
	// a step size function that returns the step size,
	// a prox operator that returns the prox value
	SparseVector sgd_step(const SparseVector & x, 
				   sp_partial_grad_fn partial_grad,
				   sp_stepsize_fn stepsize,
				   unsigned int k,
				   unsigned int n)
	{

		// get a random row
		unsigned int i=rand()%(n-1);

		// get gradient direction
		SparseVector pgrad = partial_grad(x, i);

		// get stepsize
		double gamma = stepsize(k, x, -1*pgrad, pgrad);

		// take step
		SparseVector xnew = x - gamma*pgrad;
		return xnew;
	}


	// one step of stochastic variance reduced gradient descent
	// given the previous step x,
	// an oracle function that returns the gradient at a point x,
	// a step size function that returns the step size,
	// a prox operator that returns the prox value
	Vector svrg_step(const Vector & x, 
					 const Vector & x_old,
					 partial_grad_fn partial_grad,
					 const Vector & old_grad,
					 stepsize_fn stepsize,
					 unsigned int k,
					 unsigned int n)
	{

		// get a random row
		unsigned int i=rand()%(n-1);

		// get gradient direction
		Vector pgrad = partial_grad(x, i);
		Vector pgrad_old = partial_grad(x_old, i);
		Vector gd = pgrad - pgrad_old + old_grad;
		// Vector gd = pgrad;


		// get stepsize
		double gamma = stepsize(k, x, -1*gd, gd);
		
		// take step
		Vector xnew = x - gamma*gd;
		return xnew;
	}


	// one step of sparse stochastic variance reduced gradient descent (KroMagnon)
	// given the previous step x,
	// an oracle function that returns the gradient at a point x,
	// a step size function that returns the step size,
	// a prox operator that returns the prox value
	SparseVector svrg_step(const SparseVector & x, 
					 const SparseVector & x_old,
					 sp_partial_grad_fn partial_grad,
					 const Vector & old_grad,
					 sp_stepsize_fn stepsize,
					 unsigned int k,
					 unsigned int n)
	{

		// get a random row
		unsigned int i=rand()%(n-1);

		// get gradient direction
		SparseVector pgrad = partial_grad(x, i);
		SparseVector pgrad_old = partial_grad(x_old, i);
		SparseVector sp_old_grad = old_grad.get_support(pgrad);
		SparseVector gd = (pgrad - pgrad_old + sp_old_grad);


		// get stepsize
		double gamma = stepsize(k, x, -1*gd, gd);
		
		// take step
		SparseVector xnew = x - gamma*gd;
		return xnew;
	}




	// frank-wolfe step
	Vector fw_step(const Vector & x, 
					 oracle_fn smin,
					 stepsize_fn stepsize,
					 unsigned int k)
	{

		// get the direction s
		Vector s = smin(x);
		Vector d = x-s;

		// get stepsize
		double gamma = stepsize(k, x, -1*d, d);

		// take step
		Vector xnew = x - gamma*d;

		return xnew;
	}


	/*
	// sparse frank-wolfe step
	SparseVector fw_step(const SparseVector & x, 
					 oracle_fn grad,
					 sp_frankwolfe s,
					 sp_stepsize_fn stepsize,
					 unsigned int k)
	{
		// get the gradient

		// get the direction s

		// get stepsize

		// take step
		SparseVector xnew = (1-gamma)*x + gamma*s;
	}

	/*
	// ISTA step
	Vector ista_step(const Vector & x, 
					 const Vector & x_old,
					 partial_grad_fn partial_grad,
					 const Vector & old_grad,
					 stepsize_fn stepsize,
					 unsigned int k,
					 unsigned int n){

	}
	*/

	// Fast-ISTA (FISTA) step
	Vector fista_step(const Vector & x,
					  Vector & y,
					  double & lam,
					  oracle_fn oracle,
					  stepsize_fn stepsize,
					  proximal_fn prox,
					  unsigned int k){
		
		// get gradient direction
		Vector grad = oracle(x);
		
		// get stepsize
		double gamma = stepsize(k, x, -1*grad, grad);

		// update lambda
		double lamnew = (1.0+sqrt(1.0+4.0*lam*lam))*0.5;
		double kappa = (1-lam)/lamnew;


		// take step
		Vector xs = x - gamma*grad;
		Vector ynew = prox(xs);
		Vector xnew = (1-kappa)*ynew + kappa*y;

		// update outputs
		swap(ynew, y);
		lam = lamnew;

		return xnew;
	}
	

}


#endif