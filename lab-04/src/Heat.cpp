#include "Heat.hpp"

void
Heat::setup()
{
  pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    // Read serial mesh.
    Triangulation<dim> mesh_serial;

    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(mesh_serial);

      std::ifstream mesh_file(mesh_file_name);
      grid_in.read_msh(mesh_file);
    }

    // Copy the serial mesh into the parallel one.
    {
      GridTools::partition_triangulation(mpi_size, mesh_serial);

      const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
      mesh.create_triangulation(construction_data);
    }

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    pcout << "  Initializing the sparsity pattern" << std::endl;
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity);

    // Vector with ghost elements (read only - access to neighbors).
    pcout << "  Initializing vectors" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    // Vector for the solver (write access - no ghost).
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void
Heat::assemble()
{
  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  // Indices of the vector.
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  // 1. Prepare vectors store u^n values and gradients.
  std::vector<double>         solution_old_values(n_q);
  std::vector<Tensor<1, dim>> solution_old_grads(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      // 2. Extract u^n (old solution) values onto quadrature points.
      // FIX: Use 'solution', not 'solution_old'
      fe_values.get_function_values(solution, solution_old_values);
      fe_values.get_function_gradients(solution, solution_old_grads);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double mu_loc = mu(fe_values.quadrature_point(q));

          // Force terms f(t^n) and f(t^{n+1})
          const double f_old_loc =
            f(fe_values.quadrature_point(q), time - delta_t);
          const double f_new_loc = f(fe_values.quadrature_point(q), time);

          const double dx = fe_values.JxW(q);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const double         phi_i      = fe_values.shape_value(i, q);
              const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

              // --- MATRIX ASSEMBLY (LHS) ---
              // (1/dt) * M + theta * A.
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const double         phi_j      = fe_values.shape_value(j, q);
                  const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

                  double mass_term = (1.0 / delta_t) * phi_i * phi_j;
                  double stiff_term =
                    theta * mu_loc * (grad_phi_i * grad_phi_j);

                  cell_matrix(i, j) += (mass_term + stiff_term) * dx;
                }

              // --- MATRIX ASSEMBLY (RHS) ---

              // Term 1: Mass part (u^n / dt)
              double rhs_mass =
                (1.0 / delta_t) * phi_i * solution_old_values[q];

              // Term 2: Stiffness part - (1-theta) * mu * grad(u^n) * grad(v)
              double rhs_stiff =
                -(1.0 - theta) * mu_loc * (grad_phi_i * solution_old_grads[q]);

              // Term 3: Forcing part
              double rhs_force =
                (theta * f_new_loc + (1.0 - theta) * f_old_loc) * phi_i;

              cell_rhs(i) += (rhs_mass + rhs_stiff + rhs_force) * dx;
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Homogeneous Neumann boundary conditions: we do nothing.
}

void
Heat::solve_linear_system()
{
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  // Solve for the owned DoFs.
  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << solver_control.last_step() << " CG iterations" << std::endl;

  // Distribute the result to the ghosted vector for Output.
  solution = solution_owned;
}

void
Heat::output() const
{
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");

  // Add vector for parallel partition.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  data_out.write_vtu_with_pvtu_record(/* folder = */ "./",
                                      /* basename = */ output_file_name,
                                      /* index = */ timestep_number,
                                      MPI_COMM_WORLD);
}

void
Heat::run()
{
  // Build meshes and matrices structures
  setup();

  // 1. Initial Condition
  pcout << "Setting initial condition..." << std::endl;

  // Use the nested class we defined in the header.
  VectorTools::interpolate(dof_handler, FunctionU0(), solution_owned);

  // Copy owned values to ghosted vector 'solution' so we can read them in
  // assemble().
  solution = solution_owned;

  // Sets up the time.
  time            = 0.0;
  timestep_number = 0;
  // FIX: removed 'time_step = ...' because delta_t is const and already set.

  // Output initial state
  output();

  pcout << "===============================================" << std::endl;

  // 2. Time Loop
  pcout << "Starting the loop..." << std::endl;
  while (time < T - 0.5 * delta_t)
    {
      time += delta_t;
      ++timestep_number;
      pcout << "Timestep " << timestep_number << ", time =" << time
            << std::endl;

      assemble();
      solve_linear_system();

      // FIX: Update 'solution' (ghosted) with 'solution_owned'
      solution = solution_owned;

      output();
    }
}