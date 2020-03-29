
#pragma once

#include <redGrapes/access/io.hpp>
#include <pmacc/memory/buffers_new/hostBuffer/Data.hpp>
#include <pmacc/memory/buffers_new/common/Size.hpp>

namespace pmacc
{
namespace mem
{
namespace host_buffer
{

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
struct HostBufferResource
    : protected rg::SharedResourceObject<
          HostBufferData< T_Item, T_dim >,
          T_DataAccessPolicy
      >
{
    using Item = T_Item;
    static constexpr std::size_t dim = T_dim;
    using DataAccessPolicy = T_DataAccessPolicy;
    using DataBoxType = DataBox< PitchedBox< Item, dim > >;

    using Data = host_buffer::HostBufferData< Item, dim >;
    using Size = buffer::BufferSize< dim >;

    HostBufferResource( DataSpace< dim > capacity )
        : rg::SharedResourceObject<
              Data,
              DataAccessPolicy
          >( std::make_shared< Data >( capacity ))
    {}

    auto make_guard()
    {
        return buffer::GuardBase<HostBufferResource>( *this );
    }
};


template<
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
using ReadGuard =
    buffer::ReadGuard<
        HostBufferResource<
            T_Item,
            T_dim,
            T_DataAccessPolicy
        >
    >;

template<
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
using WriteGuard =
    buffer::WriteGuard<
        HostBufferResource<
            T_Item,
            T_dim,
            T_DataAccessPolicy
        >
    >;

} // namespace host_buffer

} // namespace mem

} // namespace pmacc

