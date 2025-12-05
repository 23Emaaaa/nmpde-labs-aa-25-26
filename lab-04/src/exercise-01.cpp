#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Define coefficients.
  const auto mu = [](const Point<dim> & /*p*/) { return 0.1; };
  const auto f  = [](const Point<dim>  &/*p*/, const double  &/*t*/) {
    return 0.0;
  };

  // Initialize problem with Exercise 1.4 settings:
  // T = 1.0
  // Theta = 1.0 (Implicit Euler)
  // Delta_t = 0.05
  Heat problem(/*mesh_filename = */ "../mesh/mesh-cube-10.msh",
               1,      // degree
               1.0,    // T
               0.0,    // theta (0.0 = Explicit Euler, 1.0 = Implicit Euler)
               0.0025, // delta_t (Smaller step needed for Explicit stability!)
               mu,
               f);

  problem.run();

  return 0;
}