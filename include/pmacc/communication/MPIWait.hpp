
#pragma once

#include <rmngr/task.hpp>
#include <pmacc/communication/MPITask.hpp>
#include <mpi.h>

namespace pmacc
{
namespace communication
{

class TaskMPITest
    : public rmngr::Task<
        TaskMPITest,
        boost::mpl::vector<MPITask>,
        bool
    >
{
protected:
    MPI_Request * request;
    MPI_Status * status;

public:
    TaskMPITest( MPI_Request * request, MPI_Status * status )
        : request(request), status(status)
    {}

    bool run()
    {
        int flag = 0;
        MPI_CHECK(MPI_Test(this->request, &flag, this->status));
	return bool(flag);
    }
};

class TaskMPIWait
    : public rmngr::Task<
        TaskMPIWait,
        boost::mpl::vector<MPITask>,
        MPI_Status
    >
{
protected:
    MPI_Request * request;

public:
    TaskMPIWait( MPI_Request * request )
        : request( request )
    {}

    MPI_Status run()
    {
        MPI_Status status;
        bool finished = false;
        while( !finished )
            finished = TaskMPITest::create( Scheduler::getInstance(), this->request, &status ).get();
        return status;
    }
};

} // namespace communication

} // namespace pmacc

