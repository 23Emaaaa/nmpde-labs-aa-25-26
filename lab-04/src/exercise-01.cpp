#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = Heat::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Define coefficients.
  const auto mu = [](const Point<dim> & /*p*/) -> double { return 0.1; };
  const auto f  = [](const Point<dim>  &/*p*/, const double  &/*t*/) -> double {
    return 0.0;
  };

  // Initialize problem with Exercise 1.4 settings:
  // T = 1.0
  // Theta = 1.0 (Implicit Euler)
  // Delta_t = 0.05
  Heat problem(/*mesh_filename = */ "../mesh/mesh-cube-10.msh",
               /* degree = */ 1,
               /* T = */ 1.0,
               /* theta = */ 1.0,
               /* delta_t = */ 0.05,
               mu,
               f);

  problem.run();

  return 0;
}