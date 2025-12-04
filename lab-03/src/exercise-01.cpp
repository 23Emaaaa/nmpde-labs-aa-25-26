#include "DiffusionReaction.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  // 1. Define the Problem Parameters.
  // ---------------------------------

  constexpr unsigned int dim = DiffusionReaction::dim;

  // The mesh file to use, respectively:
  // Lab Task 1.1 asks for "mesh/mesh-cube-10.msh".
  // Lab Task 1.2 asks for "mesh/mesh-cube-40.msh".
  const std::string mesh_file_name = "../mesh/mesh-cube-40.msh";

  // Polynomial degree "r".
  const unsigned int r = 1;

  // 2. Define the coefficients (Lambda Functions).
  // ----------------------------------------------

  // Diffusion Coefficient mu(x):
  // mu = 100 if x < 1/2 or mu = 1 if x >= 1/2
  // Note: we use 1/2 = 0.5
  const auto mu = [](const Point<dim> &p) -> double {
    if (p[0] < 0.5)
      return 100.0;
    else
      return 1.0;
  };

  // Reaction Coefficient (constant) sigma(x) = 1.
  const auto sigma = [](const Point<dim> & /*p*/) -> double { return 1.0; };

  // Forcing coefficient (constant) f(x) = 1.
  const auto f = [](const Point<dim> & /*p*/) -> double { return 1.0; };

  // 3. Initialize and Run the Solver.
  // ---------------------------------

  // Instantiate the problem class.
  DiffusionReaction problem(mesh_file_name, r, mu, sigma, f);

  // Runt the standard FEM lifecycle.
  problem.setup();    // Create the mesh, DoFs, Matrices.
  problem.assemble(); // Calculates integrals (LHS and RHS).
  problem.solve();    // Solve Ax = b
  problem.output();   // Write .vtk file for ParaView visualization.

  return 0;
}