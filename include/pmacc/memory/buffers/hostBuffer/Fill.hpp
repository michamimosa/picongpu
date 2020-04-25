
#pragma once

#include <pmacc/Environment.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/memory/buffers/hostBuffer/Resource.hpp>

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
    host_buffer::WriteGuard<
        T_Item,
        T_dim,
        T_DataAccessPolicy
    > const & host_buffer,
    T_Item value
)
{
    return Environment<>::task(
        [value]( auto host_buffer )
        {
            int64_t current_size = static_cast< int64_t >( host_buffer.size().getCurrentSize() );
            DataBoxDim1Access< DataBox< PitchedBox< T_Item, T_dim > > > d1Box(
                host_buffer.data().getDataBox(),
                host_buffer.size().getDataSpace()
            );

            #pragma omp parallel for
            for (int64_t i = 0; i < current_size; i++)
                d1Box[i] = value;
        },
        TaskProperties::Builder()
            .label("fill HostBuffer"),
        host_buffer.write()
    );
}

} // namespace buffer

} // namespace mem

} // namespace pmacc

