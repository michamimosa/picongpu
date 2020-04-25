
#pragma once

#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/memory/buffers/Buffer.hpp>
#include <pmacc/memory/buffers/hostBuffer/Resource.hpp>
#include <pmacc/memory/buffers/Reset.hpp>

namespace pmacc
{
namespace mem
{

template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy = rg::access::IOAccess
>
struct HostBuffer
    : host_buffer::WriteGuard<
        T_Item,
        T_dim,
        T_DataAccessPolicy
    >
{
    /*! create a new host buffer
     *
     * @param capacity extent for each dimension (in elements)
     */
    HostBuffer( DataSpace< T_dim > capacity )
        : host_buffer::WriteGuard<
              T_Item,
              T_dim,
              T_DataAccessPolicy
          >(
              host_buffer::HostBufferResource<
                  T_Item,
                  T_dim,
                  T_DataAccessPolicy
              >( capacity )
              .make_guard()
          )
    {
        pmacc::mem::buffer::reset( *this, false );
    }
};

} // namespace mem

} // namespace pmacc

