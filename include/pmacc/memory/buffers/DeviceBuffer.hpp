
#pragma once

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/memory/buffers/Buffer.hpp>
#include <pmacc/memory/buffers/Reset.hpp>
#include <pmacc/memory/buffers/deviceBuffer/Data.hpp>
#include <pmacc/memory/buffers/deviceBuffer/Size.hpp>
#include <pmacc/memory/buffers/deviceBuffer/Resource.hpp>

namespace pmacc
{
namespace mem
{

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
struct DeviceBuffer
    : device_buffer::WriteGuard<
        T_Item,
        T_dim,
        T_DataAccessPolicy
    >
{
    /*! create a new device buffer
     *
     * @param capacity extent for each dimension (in elements)
     * @param size_on_device whether a copy of the size is stored on device
     * @param use_vector_as_base use a vector as base of the array (is not lined pitched)
     *                           if true size_on_device is atomaticly set to false
     */
    DeviceBuffer(
        DataSpace< T_dim > capacity,
        bool size_on_device = false,
        bool use_vector_as_base = false
    ) :
        device_buffer::WriteGuard<
            T_Item,
            T_dim,
            T_DataAccessPolicy
        >(
            device_buffer::DeviceBufferResource<
                T_Item,
                T_dim,
                T_DataAccessPolicy
            >(
                capacity,
                use_vector_as_base
            ).make_guard( size_on_device )
        )
    {
        buffer::reset( *this, false );
    }
};

} // namespace mem

} // namespace pmacc

