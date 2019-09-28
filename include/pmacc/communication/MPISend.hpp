
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/communication/ICommunicator.hpp>
#include <pmacc/communication/MPIWait.hpp>
#include <pmacc/memory/buffers/Exchange.hpp>
#include <pmacc/memory/buffers/CopyDeviceToHost.hpp>
#include <pmacc/memory/buffers/CopyDeviceToDevice.hpp>
#include <pmacc/Environment.hpp>

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
    return Environment<>::get().ResourceManager().emplace_task(
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
        TaskProperties::Builder()
            .label("mpi_send(exchange_type = " + std::to_string(exchange_type) + ", communication_tag = " + std::to_string(communication_tag) + ")")
            .mpi_task()
            .resources({
                host_buffer.read(),
                host_buffer.size_resource.read()
            })
    );
};

} // namespace communication

} // namespace pmacc

