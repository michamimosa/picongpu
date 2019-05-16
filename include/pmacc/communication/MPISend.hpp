
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/communication/ICommunicator.hpp>
#include <pmacc/communication/Tasks.hpp>
#include <pmacc/communication/MPITask.hpp>
#include <pmacc/communication/MPIWait.hpp>
#include <pmacc/memory/buffers/Exchange.hpp>
#include <pmacc/memory/buffers/CopyDeviceToHost.hpp>
#include <pmacc/memory/buffers/CopyDeviceToDevice.hpp>

#include <mpi.h>

namespace pmacc
{
namespace communication
{

struct MPISendLabel
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
class TaskMPISend
  : public rmngr::Task<
               TaskSendMPI<T, T_Dim>,
               boost::mpl::vector<
        	   SendTask<T, T_Dim>,
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
        if (this->exchange->hasDeviceDoubleBuffer())
	{
            memory::buffers::TaskCopyDeviceToDevice<T, T_Dim>::create(
                Scheduler::getInstance(),
		this->exchange->getDeviceBuffer(),
                this->exchange->getDeviceDoubleBuffer());
            memory::buffers::TaskCopyDeviceToHost<T, T_Dim>::create(
                Scheduler::getInstance(),
		this->exchange->getDeviceDoubleBuffer(),
                this->exchange->getHostBuffer());
	}
	else
        {
            memory::buffers::TaskCopyDeviceToHost<T, T_Dim>::create(
                Scheduler::getInstance(),
	        this->exchange->getDeviceBuffer(),
                this->exchange->getHostBuffer());
	}

	Scheduler::getInstance().update_property< rmngr::ResourceUserPolicy >(
									     {
	      this->exchange->getHostBuffer().write(),
	      this->exchange->getHostBuffer().size_resource.write() });

	MPI_Request * request = Environment<T_Dim>::get()
	  .EnvironmentController()
	  .getCommunicator().startSend(
	      this->exchange->getExchangeType(),
	      (char*) this->exchange->getHostBuffer().getPointer(),
	      this->exchange->getHostBuffer().getCurrentSize() * sizeof (T),
	      this->exchange->getCommunicationTag());

        TaskMPIWait::create( Scheduler::getInstance(), request );
    }
};

} // namespace communication

} // namespace pmacc

