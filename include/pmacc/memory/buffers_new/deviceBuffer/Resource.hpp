
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
    typename T_DataAccessPolicy
>
struct DeviceBufferResource;



namespace data
{

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
struct ReadGuard
    : pmacc::mem::buffer::data::ReadGuard<
        DeviceBufferResource< T_Item, T_dim, T_DataAccessPolicy >
    >
{
    cudaPitchedPtr getCudaPitched() const { return this->data.obj->get_cuda_pitched();  }

    ReadGuard read() const noexcept { return *this; }

    ReadGuard( buffer::GuardBase< DeviceBufferResource< T_Item, T_dim, T_DataAccessPolicy > > const & base )
        : pmacc::mem::buffer::data::ReadGuard< DeviceBufferResource<T_Item, T_dim, T_DataAccessPolicy> >( base )
    {}
};

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
struct WriteGuard
    : pmacc::mem::buffer::data::WriteGuard<
        DeviceBufferResource< T_Item, T_dim, T_DataAccessPolicy >
    >
{
    cudaPitchedPtr getCudaPitched() const { return this->data.obj->get_cuda_pitched();  }

    ReadGuard< T_Item, T_dim, T_DataAccessPolicy > read() const noexcept { return *this; }
    WriteGuard write() const noexcept { return *this; }

    WriteGuard( buffer::GuardBase< DeviceBufferResource< T_Item, T_dim, T_DataAccessPolicy > > const & base )
        : pmacc::mem::buffer::data::WriteGuard< DeviceBufferResource<T_Item, T_dim, T_DataAccessPolicy> >( base )
    {}
};

} // namespace data

namespace size
{
template<
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
using ReadGuard =
    buffer::size::ReadGuard<
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
    buffer::size::WriteGuard<
        DeviceBufferResource<
            T_Item,
            T_dim,
            T_DataAccessPolicy
        >
    >;
} // namespace size



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

    using DataGuard = data::WriteGuard< Item, dim, DataAccessPolicy >;
    using SizeGuard = size::WriteGuard< Item, dim, DataAccessPolicy >;

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



namespace redGrapes
{
namespace trait
{

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
struct BuildProperties<
    pmacc::mem::device_buffer::size::ReadGuard< T_Item, T_dim, T_DataAccessPolicy >
>
{
    template < typename Builder >
    static void build(Builder & builder, pmacc::mem::device_buffer::size::ReadGuard< T_Item, T_dim, T_DataAccessPolicy > const & buf)
    {
        if( buf.size.device_current_size )
        {
            builder.add( buf.size.host_current_size.write() );
            builder.add( buf.size.device_current_size->read() );

            // fixme
            builder.add( pmacc::Environment<>::get().cuda_stream() );
        }
        else
            builder.add( buf.size.host_current_size.read() );
    }
};

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
struct BuildProperties<
    pmacc::mem::device_buffer::size::WriteGuard< T_Item, T_dim, T_DataAccessPolicy >
>
{
    template < typename Builder >
    static void build(Builder & builder, pmacc::mem::device_buffer::size::WriteGuard< T_Item, T_dim, T_DataAccessPolicy > const & buf)
    {
        builder.add( buf.size.host_current_size.write() );

        if( buf.size.device_current_size )
        {
            builder.add( buf.size.device_current_size->write() );

            // fixme
            builder.add( pmacc::Environment<>::get().cuda_stream() );
        }
    }
};

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
struct BuildProperties<
    pmacc::mem::device_buffer::data::ReadGuard< T_Item, T_dim, T_DataAccessPolicy >
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::device_buffer::data::ReadGuard< T_Item, T_dim, T_DataAccessPolicy > const & buf
    )
    {
        builder.add( (pmacc::mem::buffer::data::ReadGuard< pmacc::mem::device_buffer::DeviceBufferResource< T_Item, T_dim, T_DataAccessPolicy > > const &) buf );
    }
};

template < typename T_Item, std::size_t T_dim, typename T_DataAccessPolicy >
struct BuildProperties<
    pmacc::mem::device_buffer::data::WriteGuard< T_Item, T_dim, T_DataAccessPolicy >
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::device_buffer::data::WriteGuard< T_Item, T_dim, T_DataAccessPolicy > const & buf
    )
    {
        builder.add( (pmacc::mem::buffer::data::WriteGuard< pmacc::mem::device_buffer::DeviceBufferResource< T_Item, T_dim, T_DataAccessPolicy > > const &) buf );
    }
};

} // namespace trait

} // namespace redGrapes


