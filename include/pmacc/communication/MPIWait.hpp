
#pragma once

#include <pmacc/type/Scheduler.hpp>
#include <mpi.h>

namespace pmacc
{
namespace communication
{

auto task_mpi_test( MPI_Request * request, MPI_Status * status )
{
    Scheduler::Properties prop;
    prop.policy< rmngr::DispatchPolicy<PMaccDispatch> >().job_selector_prop.mpi_thread = true;
    prop.policy< GraphvizPolicy >().label = "task_mpi_test()";

    return Scheduler::emplace_task(
        [request, status]
        {
            int flag = 0;
            MPI_CHECK(MPI_Test(request, &flag, status));
            return bool(flag);
        },
        prop
    );
}

auto task_mpi_wait( MPI_Request * request )
{
    Scheduler::Properties prop;
    prop.policy< GraphvizPolicy >().label = "task_mpi_wait()";

    return Scheduler::emplace_task(
        [request]
        {
            MPI_Status status;
            while( ! (task_mpi_test(request, &status).get()) );
            return status;
        },
        prop
    );
}

} // namespace communication

} // namespace pmacc

