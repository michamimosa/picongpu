
#include <pmacc/communication/MPIWait.hpp>
#include <pmacc/communication/manager_common.hpp>
#include <pmacc/communication/ICommunicator.hpp>
#include <pmacc/memory/buffers/Exchange.hpp>
#include <pmacc/memory/buffers/CopyHostToDevice.hpp>
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
task_mpi_receive(
    HostBuffer<T, T_Dim > & host_buffer,
    uint32_t exchange_type,
    uint32_t communication_tag
)
{
    return Environment<>::get().ResourceManager().emplace_task(
        [&host_buffer, exchange_type, communication_tag]
        {
            MPI_Request * request =
                Environment<>::get().ResourceManager().emplace_task(
                    [&host_buffer, exchange_type, communication_tag]
                    {
                        return Environment< T_Dim >::get()
                                  .EnvironmentController()
                                  .getCommunicator()
                                  .startReceive(
                                      exchange_type,
                                      ( char * ) host_buffer.getBasePointer(),
                                      host_buffer.getDataSpace().productOfComponents() * sizeof( T ),
                                      communication_tag
                                  );
                    },
                    TaskProperties::Builder()
                        .label("MPI request")
                        .mpi_task()
                        .resources({
                            host_buffer.write(),
                            host_buffer.size_resource.read()
                        })
                ).get();

            auto s = task_mpi_wait(request);
            rmngr::IOResource status_resource = s.first;
            std::shared_ptr<MPI_Status> status = s.second;

            Environment<>::get().ResourceManager().emplace_task(
                [status, communication_tag, request, &host_buffer]
                {
                    int recv_data_count = Environment<>::get().ResourceManager().emplace_task(
                         [status]
                         {
                             int recv_data_count;
                             MPI_CHECK_NO_EXCEPT( MPI_Get_count( status.get(), MPI_CHAR, &recv_data_count ) );
                             return recv_data_count;
                         },
                         TaskProperties::Builder()
                            .label("MPI_Get_count()")
                            .mpi_task()
                    ).get();

                    if( recv_data_count == MPI_UNDEFINED )
                        std::cerr << "undefined number of elements received" << std::endl;

                    size_t newBufferSize = recv_data_count / sizeof( T );
                    //std::cout << "received " << newBufferSize << " elements" << std::endl;

                    host_buffer.setCurrentSize( newBufferSize );
                },
                TaskProperties::Builder()
                    .label("update size")
                    .resources({
                        status_resource.read(),
                        host_buffer.write(),
                        host_buffer.size_resource.write()
                    })
            );
        },
        TaskProperties::Builder()
            .label("mpi_receive(exchange_type = " + std::to_string(exchange_type) + ", communication_tag = " + std::to_string(communication_tag) + ")")
            .resources({
                host_buffer.write(),
                host_buffer.size_resource.write()
            })
    );
}

} // namespace communication

} // namespace pmacc

