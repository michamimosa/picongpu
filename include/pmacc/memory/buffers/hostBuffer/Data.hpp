
#pragma once

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>

#include <pmacc/memory/buffers/Buffer.hpp>
#include <pmacc/memory/buffers/common/Size.hpp>

namespace pmacc
{
namespace mem
{
namespace host_buffer
{

template <
    typename T_Item,
    std::size_t T_dim
>
struct HostBufferData
    : buffer::BufferData< T_Item, T_dim >
{
    using Item = T_Item;
    static constexpr std::size_t dim = T_dim;
    using DataBoxType = DataBox< PitchedBox< Item, dim > >;

    HostBufferData( DataSpace< dim > capacity )
        : capacity( capacity )
    {
        CUDA_CHECK(cudaMallocHost(
            (void**) &ptr,
            capacity.productOfComponents() * sizeof( Item )
        ));
    }

    ~HostBufferData()
    {
        CUDA_CHECK_NO_EXCEPT(cudaFreeHost(this->ptr));
    }

    DataSpace< dim > get_capacity() const noexcept
    {
        return capacity;
    }

    Item * get_base_ptr() const noexcept
    {
        return ptr;
    }

    DataBoxType get_data_box( DataSpace< dim > offset ) const noexcept
    {
        return DataBoxType(
                   PitchedBox< Item, dim >(
                       get_base_ptr(),
                       offset,
                       capacity,
                       capacity[0] * sizeof (Item)
                   )
               );
    }

    Item * get_pointer( DataSpace< dim > offset ) const noexcept
    {
        return &(*get_data_box(offset));
    }

    std::size_t getPitch() const noexcept
    {
        return capacity[0] * sizeof( Item );
    }

protected:
    Item * ptr;
    DataSpace< dim > capacity;
};

} // namespace host_buffer

} // namespace mem

} // namespace pmacc

