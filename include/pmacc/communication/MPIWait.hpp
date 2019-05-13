
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

public:
    TaskMPITest( MPI_Request * request )
        : request(request)
    {}

    bool run()
    {
        int flag = 0;
	MPI_Status status;
        MPI_CHECK(MPI_Test(this->request, &flag, &status));
	return bool(flag);
    }
};

class TaskMPIWait
    : public rmngr::Task<
        TaskMPIWait,
        boost::mpl::vector<MPITask>
    >
{
protected:
    MPI_Request * request;

public:
    TaskMPIWait( MPI_Request * request )
        : request( request )
    {}

    void run()
    {
        bool finished = false;
        while( !finished )
	{
             finished = TaskMPITest::create( Scheduler::getInstance(), this->request ).get();
	}
    }
};

} // namespace communication

} // namespace pmacc

