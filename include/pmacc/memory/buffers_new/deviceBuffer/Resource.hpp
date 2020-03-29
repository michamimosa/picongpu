
#pragma once

#include <redGrapes/access/io.hpp>
#include <redGrapes/resource/resource.hpp>

#include <pmacc/memory/buffers_new/deviceBuffer/Data.hpp>
#include <pmacc/memory/buffers_new/deviceBuffer/Size.hpp>

namespace pmacc
{
namespace mem
{
namespace device_buffer
{

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
struct DeviceBufferResource
    : protected rg::SharedResourceObject<
          DeviceBufferData< T_Item, T_dim >,
          T_DataAccessPolicy
      >
{
    using Item = T_Item;
    static constexpr std::size_t dim = T_dim;
    using DataAccessPolicy = T_DataAccessPolicy;
    using DataBoxType = DataBox< PitchedBox< Item, dim > >;

    using Data = DeviceBufferData< Item, dim >;
    using Size = DeviceBufferSize< dim >;

    DeviceBufferResource(
        DataSpace< dim > capacity,
        bool use_vector_as_base = false
    ) :
        rg::SharedResourceObject<
            Data,
            DataAccessPolicy
        >(
            std::make_shared< Data >(
                capacity,
                use_vector_as_base
            )
        ),
        use_vector_as_base( use_vector_as_base )
    {}

    auto make_guard( bool size_on_device = false )
    {
        auto guard = buffer::GuardBase<DeviceBufferResource>( *this );

        if( ! use_vector_as_base )
        {
            guard.data1D = false;

            if( size_on_device )
                guard.size.create_size_on_device();
        }

        return guard;
    }

protected:
    bool use_vector_as_base;
};


template<
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
using ReadGuard =
    buffer::ReadGuard<
        DeviceBufferResource<
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
        DeviceBufferResource<
            T_Item,
            T_dim,
            T_DataAccessPolicy
        >
    >;

} // namespace device_buffer

} // namespace mem

} // namespace pmacc


