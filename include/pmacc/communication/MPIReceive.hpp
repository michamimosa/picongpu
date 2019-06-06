
#include <pmacc/communication/MPITask.hpp>
#include <pmacc/communication/MPIWait.hpp>
#include <pmacc/communication/Tasks.hpp>
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/communication/ICommunicator.hpp>
#include <pmacc/memory/buffers/Exchange.hpp>
#include <pmacc/memory/buffers/CopyHostToDevice.hpp>
#include <pmacc/memory/buffers/CopyDeviceToDevice.hpp>

#include <mpi.h>

namespace pmacc
{
namespace communication
{

struct MPIReceiveLabel
{
    void properties(Scheduler::Schedulable& s)
    {
        s.proto_property< GraphvizPolicy >().label = "MPI Receive";
    }
};

template <
    typename T,
    size_t T_Dim
>
class TaskMPIReceive
  : public rmngr::Task<
               TaskMPIReceive<T, T_Dim>,
               boost::mpl::vector<
                   ReceiveTask<T, T_Dim>,
                   MPITask,
		   MPIReceiveLabel
	       >
           >
{
public:
    TaskMPIReceive(Exchange<T, T_Dim> & exchange)
    {
        this->exchange = &exchange;
    }

    void
    run()
    {
        MPI_Request * request =
            Environment< T_Dim >::get()
            .EnvironmentController()
            .getCommunicator()
            .startReceive( this->exchange->getExchangeType(),
                           ( char * )this->exchange->getHostBuffer()
                           .getBasePointer(),
                           this->exchange->getHostBuffer()
                           .getDataSpace()
                           .productOfComponents() *
                           sizeof( T ),
                           this->exchange->getCommunicationTag() );

        MPI_Status status = TaskMPIWait::create( Scheduler::getInstance(), request ).get();

        int recv_data_count;
        MPI_CHECK_NO_EXCEPT( MPI_Get_count( &status, MPI_CHAR, &recv_data_count ) );

        if( recv_data_count == MPI_UNDEFINED )
            std::cerr << "undefined number of elements received" << std::endl;

        size_t newBufferSize = recv_data_count / sizeof( T );
        std::cout << "received " << newBufferSize << " elements" << std::endl;
        this->exchange->getHostBuffer().setCurrentSize( newBufferSize );
    }
};

} // namespace communication

} // namespace pmacc

