
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/communication/ICommunicator.hpp>
#include <pmacc/communication/MPIWait.hpp>
#include <pmacc/memory/buffers/Exchange.hpp>
#include <pmacc/memory/buffers/CopyDeviceToHost.hpp>
#include <pmacc/memory/buffers/CopyDeviceToDevice.hpp>

#include <mpi.h>

namespace pmacc
{
namespace communication
{

template <
    typename T,
    std::size_t T_Dim
>
auto
task_mpi_send(
    HostBuffer<T, T_Dim > & host_buffer,
    uint32_t exchange_type,
    uint32_t communication_tag
)
{
    Scheduler::Properties prop;
    prop.policy< rmngr::DispatchPolicy<PMaccDispatch> >().job_selector_prop.mpi_thread = true;
    prop.policy< rmngr::ResourceUserPolicy >() += host_buffer.read();
    prop.policy< rmngr::ResourceUserPolicy >() += host_buffer.size_resource.read();
    prop.policy< GraphvizPolicy >().label = "task_mpi_send()";

    return Scheduler::emplace_task(
        [&host_buffer, exchange_type, communication_tag]
        {
            MPI_Request * request =
                Environment< T_Dim >::get()
                .EnvironmentController()
                .getCommunicator()
                .startSend(
                    exchange_type,
                    ( char * ) host_buffer.getPointer(),
                    host_buffer.getCurrentSize() * sizeof( T ),
                    communication_tag
                );

            task_mpi_wait( request );
        },
        prop
    );
};

} // namespace communication

} // namespace pmacc

