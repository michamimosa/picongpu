
#pragma once

#include <redGrapes/access/io.hpp>
#include <redGrapes/resource/resource.hpp>

#include <pmacc/memory/buffers/deviceBuffer/Data.hpp>
#include <pmacc/memory/buffers/deviceBuffer/Size.hpp>

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
struct WriteGuard;

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
    cuplaPitchedPtr getCudaPitched() const { return this->data.obj->get_cuda_pitched();  }

    ReadGuard< T_Item, T_dim, T_DataAccessPolicy > read() const noexcept { return *this; }

    ReadGuard( buffer::GuardBase< DeviceBufferResource< T_Item, T_dim, T_DataAccessPolicy > > const & base ) noexcept
        : buffer::data::ReadGuard< DeviceBufferResource<T_Item, T_dim, T_DataAccessPolicy> >( base )
    {}

    ReadGuard( WriteGuard< T_Item, T_dim, T_DataAccessPolicy > const & wr ) noexcept
        : buffer::data::ReadGuard< DeviceBufferResource<T_Item, T_dim, T_DataAccessPolicy> >( (buffer::data::WriteGuard<DeviceBufferResource<T_Item, T_dim, T_DataAccessPolicy>> const&) wr )
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
    cuplaPitchedPtr getCudaPitched() const { return this->data.obj->get_cuda_pitched();  }

    ReadGuard< T_Item, T_dim, T_DataAccessPolicy > read() const noexcept { return ReadGuard< T_Item, T_dim, T_DataAccessPolicy >(*this); }
    WriteGuard< T_Item, T_dim, T_DataAccessPolicy > write() const noexcept { return *this; }

    WriteGuard( buffer::GuardBase< DeviceBufferResource< T_Item, T_dim, T_DataAccessPolicy > > const & base ) noexcept
        : buffer::data::WriteGuard< DeviceBufferResource<T_Item, T_dim, T_DataAccessPolicy> >( base )
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

    using DataGuard = device_buffer::data::WriteGuard< Item, dim, DataAccessPolicy >;
    using SizeGuard = device_buffer::size::WriteGuard< Item, dim, DataAccessPolicy >;

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

    DeviceBufferResource(
        buffer::GuardBase< DeviceBufferResource > const & other,
        bool use_vector_as_base = false        
    ) :
        rg::SharedResourceObject<
            Data,
            DataAccessPolicy
        >( other.data ),
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



namespace buffer
{
namespace size
{

template<
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
struct ReadGuard<
    device_buffer::DeviceBufferResource<
        T_Item,
        T_dim,
        T_DataAccessPolicy
    >
>
    : GuardBase<
        device_buffer::DeviceBufferResource<
            T_Item,
            T_dim,
            T_DataAccessPolicy
        >
    >
{
    ReadGuard( GuardBase< device_buffer::DeviceBufferResource< T_Item, T_dim, T_DataAccessPolicy > > const & base )
        : GuardBase< device_buffer::DeviceBufferResource<T_Item, T_dim, T_DataAccessPolicy> >( base )
    {}

    std::vector< rg::ResourceAccess > get_access() const
    {
        std::vector< rg::ResourceAccess > acc;
        this->size.push_read_access( acc );
        return acc;
    }

    ReadGuard read() { return *this; }

    std::size_t get() const { return this->size.get_current_size(); }
    std::size_t getCurrentSize() const { return this->size.get_current_size(); }
    DataSpace< T_dim > getCurrentDataSpace() const { return this->size.get_current_data_space(); }

    bool is_on_device()
    {
        return this->size.is_on_device();
    }

    size_t * get_device_pointer()
    {
        return this->size.get_device_pointer();
    }
};

} // namespace size
} // namespace buffer


} // namespace mem

} // namespace pmacc


template<
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
struct redGrapes::trait::BuildProperties<
    pmacc::mem::device_buffer::data::WriteGuard<T_Item, T_dim, T_DataAccessPolicy>
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::device_buffer::data::WriteGuard<T_Item, T_dim, T_DataAccessPolicy> const & buf
    )
    {
        for( auto acc : buf.get_access() )
            builder.add( acc );
    }
};

template<
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
struct redGrapes::trait::BuildProperties<
    pmacc::mem::device_buffer::data::ReadGuard<T_Item, T_dim, T_DataAccessPolicy>
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::device_buffer::data::ReadGuard<T_Item, T_dim, T_DataAccessPolicy> const & buf
    )
    {
        for( auto acc : buf.get_access() )
            builder.add( acc );
    }
};


