
#pragma once

#include <pmacc/assert.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>

#include <akrzemi/optional.hpp>

namespace pmacc
{
namespace mem
{
namespace device_buffer
{

template < std::size_t T_dim >
struct DeviceBufferSize
    : buffer::BufferSize< T_dim >
{
    void create_size_on_device()
    {
        size_t * device_size_ptr;
        CUDA_CHECK(cuplaMalloc( (void**)&device_size_ptr, sizeof(size_t) ));

        this->device_current_size =
            rg::IOResource< size_t >(
                std::shared_ptr< size_t >(
                    device_size_ptr,
                    []( size_t * ptr )
                    {
                        CUDA_CHECK_NO_EXCEPT(cuplaFree(ptr));
                    }));

        this->reset();
    }

    std::size_t get_current_size() const
    {
        if( device_current_size )
            Environment<>::task(
                []( auto host_size, auto device_size )
                {
                    CUDA_CHECK(cuplaMemcpyAsync(
                        host_size.get(),
                        device_size.get(),
                        sizeof( std::size_t ),
                        cuplaMemcpyDeviceToHost,
                        redGrapes::thread::current_cupla_stream
                    ));
                },

                TaskProperties::Builder()
                    .label("DeviceBufferSize: sync host size")
                    .scheduling_tags({ SCHED_CUPLA }),

                this->host_current_size.write(),
                this->device_current_size->read()
            );

        return buffer::BufferSize< T_dim >::get_current_size();
    }

    struct KernelSetValueOnDeviceMemory
    {
        template< typename T_Acc >
        DINLINE void operator()(T_Acc const &, size_t* pointer, size_t const size) const
        {
            *pointer = size;
        }
    };

    void set_current_size( std::size_t new_size ) const
    {
        buffer::BufferSize< T_dim >::set_current_size( new_size );

        if( device_current_size )
            Environment<>::task(
                [new_size]( auto host_size, auto device_size )
                {
                    CUPLA_KERNEL( KernelSetValueOnDeviceMemory )(
                        1,
                        1,
                        0,
                        redGrapes::thread::current_cupla_stream
                    )(
                        device_size.get(),
                        new_size
                    );
                },

                TaskProperties::Builder()
                    .label("DeviceBufferSize: sync device size")
                    .scheduling_tags({ SCHED_CUPLA }),

                this->host_current_size.read(),
                this->device_current_size->write()
            );
    }

protected:
    template < typename, typename >
    friend class rg::trait::BuildProperties;

    std::optional<
        rg::IOResource< size_t >
    > device_current_size;
};

} // namespace device_buffer

} // namespace mem

} // namespace pmacc


