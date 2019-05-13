
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
        s.proto_property< GraphvizPolicy >().label = "MPI Send";
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

    void run()
    {
        MPI_Request * request = Environment<T_Dim>::get()
	    .EnvironmentController()
	    .getCommunicator()
	    .startReceive(
	        this->exchange.getExchangeType(),
		(char*) this->exchange->getHostBuffer().getBasePointer(),
		this->exchange.getHostBuffer().getDataSpace().productOfComponents() * sizeof (T),
		this->exchange->getCommunicationTag());

	TaskMPIWait::create( Scheduler::getInstance(), request );

	int recv_data_count;
	MPI_Status status;
        MPI_CHECK_NO_EXCEPT(MPI_Get_count(&status, MPI_CHAR, &recv_data_count));

	size_t newBufferSize = recv_data_count / sizeof (T);
	this->exchange->getHostBuffer().setCurrentSize(newBufferSize);

	if (this->exchange->hasDeviceDoubleBuffer())
	{
            memory::buffers::TaskCopyHostToDevice<T, T_Dim>::create(
                Scheduler::getInstance(),
                this->exchange->getHostBuffer(),
                this->exchange->getDeviceDoubleBuffer());
            memory::buffers::TaskCopyDeviceToDevice<T, T_Dim>::create(
                Scheduler::getInstance(),
                this->exchange->getDeviceDoubleBuffer(),
                this->exchange->getDeviceBuffer());
	}
        else
        {
            memory::buffers::TaskCopyHostToDevice<T, T_Dim>::create(
                Scheduler::getInstance(),
                this->exchange->getHostBuffer(),
                this->exchange->getDeviceBuffer());
        }
    }
};

} // namespace communication

} // namespace pmacc

