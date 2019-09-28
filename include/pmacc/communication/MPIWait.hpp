
#pragma once

#include <pmacc/Environment.hpp>
#include <mpi.h>

namespace pmacc
{
namespace communication
{

auto task_mpi_test( MPI_Request * request, MPI_Status * status )
{
    return Environment<>::get().ResourceManager().emplace_task(
        [request, status]
        {
            int flag = 0;
            MPI_CHECK(MPI_Test(request, &flag, status));
            return bool(flag);
        },
        TaskProperties::Builder()
            .label("mpi_test()")
            .mpi_task()
    );
}

auto task_mpi_wait( MPI_Request * request )
{
    return Environment<>::get().ResourceManager().emplace_task(
        [request]
        {
            MPI_Status status;
            while( ! (task_mpi_test(request, &status).get()) );
            return status;
        },
        TaskProperties::Builder()
            .label("mpi_wait()")
    );
}

} // namespace communication

} // namespace pmacc

