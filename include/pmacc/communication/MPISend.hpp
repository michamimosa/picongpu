
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/communication/ICommunicator.hpp>
#include <pmacc/communication/Tasks.hpp>
#include <pmacc/communication/MPITask.hpp>
#include <pmacc/memory/buffers/Exchange.hpp>
#include <pmacc/memory/buffers/CopyDeviceToHost.hpp>
#include <pmacc/memory/buffers/CopyDeviceToDevice.hpp>

#include <mpi.h>

namespace pmacc {

namespace NEW {

struct MPISendLabel
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
class TaskMPISend
  : public rmngr::Task<
               TaskSendMPI,
               boost::mpl::vector<
	           SendTask,
		   MPITask,
		   MPISendLabel
	       >
           >
{
public:
    TaskMPISend(Exchange<T, T_Dim> & exchange)
    {
        this->exchange = &exchange;
    }

    void run()
    {
        if (exchange->hasDeviceDoubleBuffer())
	{
           TaskCopyDeviceToDevice<T, T_Dim>::create(
		exchange->getDeviceBuffer(),
                exchange->getDeviceDoubleBuffer());
           TaskCopyDeviceToHost<T, T_Dim>::create(
		exchange->getDeviceDoubleBuffer(),
                exchange->getHostBuffer());
	}
	else
        {
           TaskCopyDeviceToHost<T, T_Dim>::create(
	        exchange->getDeviceBuffer(),
                exchange->getHostBuffer());
	}

	MPI_Request * request = Environment<T_Dim>::get()
	  .EnvironmentController()
	  .getCommunicator().startSend(
	      exchange->getExchangeType(),
	      (char*) exchange->getHostBuffer().getPointer(),
	      exchange->getHostBuffer().getCurrentSize() * sizeof (T),
	      exchange->getCommunicationTag());

        TaskMPIWait::create( request );
    }
};

} // namespace communication

} // namespace pmacc

