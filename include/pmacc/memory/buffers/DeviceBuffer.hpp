/* Copyright 2013-2020 Heiko Burau, Rene Widera, Benjamin Worpitz
 *                     Alexander Grund, Michael Sippel
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/cuSTL/container/view/View.hpp"
#include "pmacc/cuSTL/container/DeviceBuffer.hpp"
#include "pmacc/math/vector/Int.hpp"
#include "pmacc/math/vector/Size_t.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"
#include "pmacc/types.hpp"

#include <stdexcept>

namespace pmacc
{
namespace mem
{

template <class TYPE, std::size_t DIM, typename T_DataAccessPolicy>
class Buffer;

template <class TYPE, std::size_t DIM, typename T_DataAccessPolicy>
class HostBuffer;

template <class TYPE, std::size_t DIM, typename T_DataAccessPolicy>
class DeviceBuffer;

namespace device_buffer
{
namespace data
{

template < typename Buffer >
struct ReadGuard
    : pmacc::mem::buffer::data::ReadGuard< Buffer >
{
    size_t getPitch() const { this->get()->getPitch(); }
    cudaPitchedPtr const getCudaPitched() const { return this->get()->getCudaPitched();  }

    ReadGuard read() const noexcept { return *this; }

    ReadGuard( std::shared_ptr< Buffer > const & obj )
        : pmacc::mem::buffer::data::ReadGuard< Buffer >( obj )
    {}
};

template < typename Buffer >
struct WriteGuard
    : pmacc::mem::buffer::data::WriteGuard< Buffer >
{
    size_t getPitch() const { this->get()->getPitch(); }
    cudaPitchedPtr getCudaPitched() const { return this->get()->getCudaPitched();  }

    ReadGuard<Buffer> read() const noexcept { return *this; }
    WriteGuard write() const noexcept { return *this; }

    WriteGuard( std::shared_ptr< Buffer > const & obj )
        : pmacc::mem::buffer::data::WriteGuard< Buffer >( obj )
    {}
};

} // namespace data

} // namespace device_buffer

/*
template <class TYPE, std::size_t DIM, typename T_DataAccessPolicy>
struct BufferResource< DeviceBuffer< TYPE, DIM, T_DataAccessPolicy> >
    : device_buffer::WriteGuard< DeviceBuffer<TYPE, DIM, T_DataAccessPolicy> >
{
    using Buffer = DeviceBuffer<TYPE, DIM, T_DataAccessPolicy>;

    template < typename... Args >
    BufferResource( Args&&... args )
        : device_buffer::WriteGuard<Buffer>( std::make_shared<Buffer>() )
    {
        this->get()->init( std::forward<Args>(args)... );
    }

    BufferResource( std::shared_ptr<Buffer> obj )
        : device_buffer::WriteGuard<Buffer>( obj )
    {}
};
*/
/**
 * Interface for a DIM-dimensional Buffer of type TYPE on the device.
 *
 * @tparam TYPE datatype of the buffer
 * @tparam DIM dimension of the buffer
 */
template <class TYPE, std::size_t DIM, typename T_DataAccessPolicy>
class DeviceBuffer : public Buffer<TYPE, DIM, T_DataAccessPolicy>
{
protected:
    /** constructor
     *
     * @param size extent for each dimension (in elements)
     *             if the buffer is a view to an existing buffer the size
     *             can be less than `physicalMemorySize`
     * @param physicalMemorySize size of the physical memory (in elements)
     */
    void init(DataSpace<DIM> size,
              DataSpace<DIM> physicalMemorySize)
    {
        this->Buffer<TYPE, DIM, T_DataAccessPolicy>::init( size, physicalMemorySize );
    }

public:
    //using Buffer<TYPE, DIM>::set_size; //!\todo :this function was hidden, I don't know why.

    /**
     * Destructor.
     */
    virtual ~DeviceBuffer()
    {
    };

    HINLINE
    container::CartBuffer<
        TYPE,
        DIM,
        allocator::DeviceMemAllocator<TYPE, DIM>,
        copier::D2DCopier<DIM>,
        assigner::DeviceMemAssigner<>
    >
    cartBuffer() const
    {
        cudaPitchedPtr cudaData = this->getCudaPitched();
        math::Size_t<DIM - 1> pitch;
        if(DIM >= 2)
            pitch[0] = cudaData.pitch;
        if(DIM == 3)
            pitch[1] = pitch[0] * this->getPhysicalMemorySize()[1];
        container::DeviceBuffer<TYPE, DIM> result((TYPE*)cudaData.ptr, this->getDataSpace(), false, pitch);
        return result;
    }

    /**
     * Returns offset of elements in every dimension.
     *
     * @return count of elements
     */
    virtual DataSpace<DIM> getOffset() const = 0;

    /**
     * Show if current size is stored on device.
     *
     * @return return false if no size is stored on device, true otherwise
     */
    virtual bool hasCurrentSizeOnDevice() const = 0;

    /**
     * Returns pointer to current size on device.
     *
     * @return pointer which point to device memory of current size
     */
    virtual size_t* getCurrentSizeOnDevicePointer() = 0;

    /** Returns host pointer of current size storage
     *
     * @return pointer to stored value on host side
     */
    virtual size_t* getCurrentSizeHostSidePointer() = 0;

    /**
     * Sets current size of any dimension.
     *
     * If stream is 0, this function is blocking (we use a kernel to set size).
     * Keep in mind: on Fermi-architecture, kernels in different streams may run at the same time
     * (only used if size is on device).
     *
     * @param size count of elements per dimension
     */
    virtual void set_size( size_t const size ) = 0;

    /**
     * Returns the internal pitched cuda pointer.
     *
     * @return internal pitched cuda pointer
     */
    virtual const cudaPitchedPtr getCudaPitched() const = 0;

    /** get line pitch of memory in byte
     *
     * @return size of one line in memory
     */
    virtual size_t getPitch() const = 0;

    /**
     * Copies data from the given HostBuffer to this DeviceBuffer.
     *
     * @param other the HostBuffer to copy from
     */
    //virtual void copyFrom(HostBuffer<TYPE, DIM>& other) = 0;

    /**
     * Copies data from the given DeviceBuffer to this DeviceBuffer.
     *
     * @param other the DeviceBuffer to copy from
     */
    //virtual void copyFrom(DeviceBuffer<TYPE, DIM>& other) = 0;

    device_buffer::data::WriteGuard< DeviceBuffer > data_guard()
    {
        return typename device_buffer::data::WriteGuard< DeviceBuffer >( this->template shared_from_base< DeviceBuffer >() );
    }

    rg::ResourceAccess read_size() const
    {
        // Needs to be write because the size has to be copied from device
        // TODO only write if it has to be synced, otherwise read
        return this->access_size( rg::access::IOAccess::write );
    }
};

} // namespace mem

} // namespace pmacc

namespace redGrapes
{
namespace trait
{
/*
template <
    typename T_Item,
    std::size_t T_dim,
    typename T_DataAccessPolicy
>
struct BuildProperties<
    pmacc::mem::buffer::size::ReadGuard<
        pmacc::mem::DeviceBuffer< T_Item, T_dim, T_DataAccessPolicy>
    >
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::buffer::size::ReadGuard<
            pmacc::mem::DeviceBuffer< T_Item, T_dim, T_DataAccessPolicy >
        > const & buf
    )
    {
        builder.add( buf->get()->write_size() );
    }
};
*/
template < typename Buffer >
struct BuildProperties<
    pmacc::mem::device_buffer::data::ReadGuard< Buffer >
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::device_buffer::data::ReadGuard< Buffer > const & buf
    )
    {
        builder.add( (pmacc::mem::buffer::data::ReadGuard< Buffer > const &) buf );
    }
};

template < typename Buffer >
struct BuildProperties<
    pmacc::mem::device_buffer::data::WriteGuard< Buffer >
>
{
    template < typename Builder >
    static void build(
        Builder & builder,
        pmacc::mem::device_buffer::data::WriteGuard< Buffer > const & buf
    )
    {
        builder.add( (pmacc::mem::buffer::data::WriteGuard<Buffer> const &) buf );
    }
};

} // namespace trait

} // namespace redGrapes

