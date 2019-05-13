
#include <pmacc/tasks/MPITask.hpp>
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/communication/ICommunicator.hpp>
#include <pmacc/memory/buffers/Exchange.hpp>
#include <pmacc/memory/buffers/CopyHostToDevice.hpp>
#include <pmacc/memory/buffers/CopyDeviceToDevice.hpp>

#include <mpi.h>

namespace pmacc {

namespace communication {

struct MPIReceiveLabel
{
    void properties(Scheduler::Schedulable& s)
    {
        s.proto_property< GraphvizPolicy >().label = "MPI Send";
    }
}

template <
    typename T,
    size_t T_Dim
>
class TaskMPIReceive
  : public rmngr::Task<
               TaskReceiveMPI,
               boost::mpl::vector<
		   ReceiveTask,
		   MPITask,
		   MPIReceiveLabel
	       >
           >
{
public:
    TaskReceiveMPI(Exchange<T, T_Dim> & exchange)
    {
        this->exchange = &exchange;
    }

    void run()
    {
        MPI_Request * request = Environment<T_Dim>::get()
	    .EnvironmentController()
	    .getCommunicator()
	    .startReceive(
	        exchange->getExchangeType(),
		(char*) exchange->getHostBuffer().getBasePointer(),
		exchange->getHostBuffer().getDataSpace().productOfComponents() * sizeof (T),
		exchange->getCommunicationTag());

	TaskMPIWait::create( request );

	if (data != nullptr)
	{
	    EventDataReceive *rdata = static_cast<EventDataReceive*> (data);
	    // std::cout<<" data rec "<<rdata->getReceivedCount()/sizeof(TYPE)<<std::endl;
            size_t newBufferSize = rdata->getReceivedCount() / sizeof (T);
            exchange->getHostBuffer().setCurrentSize(newBufferSize);
	}

	if (exchange->hasDeviceDoubleBuffer())
	{
            TaskCopyHostToDevice<T, T_Dim>::create(exchange->getHostBuffer(),
				       exchange->getDeviceDoubleBuffer());
            TaskCopyDeviceToDevice<T, T_Dim>::create(exchange->getDeviceDoubleBuffer(),
					   exchange->getDeviceBuffer());
	}
        else
        {
            TaskCopyHostToDevice<T, T_Dim>::create(
                exchange->getHostBuffer(),
		exchange->getDeviceBuffer());
        }
    }
};

} // namespace communication

} // namespace pmacc
