
#pragma once

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/memory/boxes/DataBox.hpp>
#include <pmacc/memory/boxes/PitchedBox.hpp>

namespace pmacc
{
namespace mem
{
namespace buffer
{

template <
    typename T_Item,
    std::size_t T_dim
>
struct BufferData
{
    using Item = T_Item;
    static constexpr std::size_t dim = T_dim;
    using DataBoxType = DataBox< PitchedBox< Item, dim > >;

    virtual ~BufferData() {};

    virtual DataSpace< dim > get_capacity() const noexcept = 0;
    virtual Item * get_base_ptr() const noexcept = 0;
    virtual Item * get_pointer( DataSpace< dim > offset ) const noexcept = 0;
    virtual DataBoxType get_data_box( DataSpace< dim > offset ) const noexcept = 0;
    virtual std::size_t getPitch() const noexcept = 0;
};

} // namespace buffer

} // namespace mem

} // namespace pmacc

