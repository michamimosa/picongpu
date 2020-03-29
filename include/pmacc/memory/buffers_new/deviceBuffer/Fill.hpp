
#pragma once

#include <pmacc/Environment.hpp>
#include <pmacc/memory/buffers_new/deviceBuffer/Resource.hpp>

namespace pmacc
{
namespace mem
{

namespace buffer
{

template<
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
auto fill(
    device_buffer::WriteGuard<
        T_Item,
        T_dim,
        T_DataAccessPolicy
    > const & device_buffer,
    T_Item value
)
{
    return Environment<>::task(
        [value]( auto device_buffer )
        {

        },
        TaskProperties::Builder()
            .label("fill DeviceBuffer"),
        device_buffer.write()
    );
}

} // namespace buffer

} // namespace mem

} // namespace pmacc



